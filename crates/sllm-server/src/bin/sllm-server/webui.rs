use std::env;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_WEBUI_PORT: u16 = 65_457;

const WEBUI_START_TIMEOUT: Duration = Duration::from_secs(20);
const WEBUI_POLL_INTERVAL: Duration = Duration::from_millis(50);
const WEBUI_STOP_TIMEOUT: Duration = Duration::from_millis(750);

pub struct WebUiProcess {
    child: Child,
    url: String,
}

impl WebUiProcess {
    pub fn start(port: u16, api_listen: SocketAddr, api_tls: bool) -> Result<Self, String> {
        ensure_port_available(port)?;
        let directory = resolve_webui_directory()?;
        let api_base_url = browser_api_base_url(api_listen, api_tls);
        let url = format!("http://localhost:{port}/");
        let mut command = Command::new("npm");
        command
            .args([
                "run",
                "dev",
                "--",
                "--hostname",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .current_dir(&directory)
            .env_clear()
            .env(
                "PATH",
                env::var_os("PATH").unwrap_or_else(|| "/usr/local/bin:/usr/bin:/bin".into()),
            )
            .env("SLLM_INTEGRATED_WEBUI", "1")
            .env("SLLM_API_BASE_URL", &api_base_url)
            .env("VINEXT_NO_DEV_LOCK", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().map_err(|error| {
            format!(
                "WebUI launch failed in {}: {error}; install its Node dependencies or start with --webui false",
                directory.display()
            )
        })?;
        let mut process = Self { child, url };
        if let Err(error) = process.wait_until_ready(port) {
            process.stop();
            return Err(error);
        }
        println!(
            "{}",
            serde_json::json!({
                "event": "webui_ready",
                "url": process.url,
                "api_base_url": api_base_url,
            })
        );
        Ok(process)
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    fn wait_until_ready(&mut self, port: u16) -> Result<(), String> {
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        let deadline = Instant::now() + WEBUI_START_TIMEOUT;
        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .map_err(|error| format!("WebUI process status check failed: {error}"))?
            {
                return Err(format!(
                    "WebUI exited before becoming ready with status {status}; start with --webui false to run API-only"
                ));
            }
            if TcpStream::connect_timeout(&address, WEBUI_POLL_INTERVAL).is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "WebUI did not become ready at {} within {} seconds",
                    self.url,
                    WEBUI_START_TIMEOUT.as_secs()
                ));
            }
            thread::sleep(WEBUI_POLL_INTERVAL);
        }
    }

    fn stop(&mut self) {
        #[cfg(unix)]
        {
            let process_group = -(self.child.id() as i32);
            let _ = self.child.try_wait();
            if !process_group_exists(process_group) {
                let _ = self.child.wait();
                return;
            }
            // The child is placed in its own process group at spawn time. This
            // stops npm, Vinext, Vite, and their local worker descendants as a
            // single unit instead of orphaning the development server.
            unsafe {
                libc::kill(process_group, libc::SIGTERM);
            }
            let deadline = Instant::now() + WEBUI_STOP_TIMEOUT;
            while Instant::now() < deadline {
                let _ = self.child.try_wait();
                if !process_group_exists(process_group) {
                    let _ = self.child.wait();
                    return;
                }
                thread::sleep(Duration::from_millis(25));
            }
            unsafe {
                libc::kill(process_group, libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
        #[cfg(not(unix))]
        {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(unix)]
fn process_group_exists(process_group: i32) -> bool {
    let status = unsafe { libc::kill(process_group, 0) };
    status == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

impl Drop for WebUiProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn browser_api_base_url(listen: SocketAddr, tls: bool) -> String {
    let browser_ip = match listen.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    let browser_address = SocketAddr::new(browser_ip, listen.port());
    let scheme = if tls { "https" } else { "http" };
    format!("{scheme}://{browser_address}")
}

fn ensure_port_available(port: u16) -> Result<(), String> {
    TcpListener::bind((Ipv4Addr::LOCALHOST, port))
        .map(drop)
        .map_err(|error| format!("WebUI port {port} is unavailable on 127.0.0.1: {error}"))
}

fn resolve_webui_directory() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Ok(executable) = env::current_exe() {
        if let Some(binary_directory) = executable.parent() {
            candidates.push(binary_directory.join("../share/sllm/webui"));
            candidates.push(binary_directory.join("../../webui"));
        }
    }
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../webui"));
    for candidate in candidates {
        if candidate.join("package.json").is_file()
            && candidate.join("node_modules/.bin/vinext").is_file()
        {
            return candidate.canonicalize().map_err(|error| {
                format!(
                    "WebUI directory {} could not be resolved: {error}",
                    candidate.display()
                )
            });
        }
    }
    Err(
        "WebUI source and installed Node dependencies were not found; run npm install in webui or start with --webui false"
            .to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_api_url_rewrites_unspecified_addresses_for_the_local_browser() {
        assert_eq!(
            browser_api_base_url("0.0.0.0:8080".parse().unwrap(), false),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            browser_api_base_url("[::]:8443".parse().unwrap(), true),
            "https://[::1]:8443"
        );
    }

    #[test]
    fn browser_api_url_preserves_an_explicit_address() {
        assert_eq!(
            browser_api_base_url("192.0.2.8:9000".parse().unwrap(), false),
            "http://192.0.2.8:9000"
        );
    }
}
