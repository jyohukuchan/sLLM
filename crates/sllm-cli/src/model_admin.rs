//! Alias-only model lifecycle administration.
//!
//! This intentionally uses a very small HTTP/1.1 client instead of accepting a
//! general URL or arbitrary request body.  Model administration is a local
//! control-plane operation: only clear-text loopback endpoints are accepted,
//! and credentials can only come from an environment variable or a file.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::Duration;

const DEFAULT_SERVER: &str = "http://127.0.0.1:8080";
const MAX_URL_BYTES: usize = 4096;
const MAX_ALIAS_BYTES: usize = 128;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Load,
    Preload,
    Unload,
    ClearQuarantine,
    EvictIdle,
}

impl Action {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "load" => Ok(Self::Load),
            "preload" => Ok(Self::Preload),
            "unload" => Ok(Self::Unload),
            "clear-quarantine" => Ok(Self::ClearQuarantine),
            "evict-idle" => Ok(Self::EvictIdle),
            _ => Err(format!("unknown model action `{value}`")),
        }
    }

    fn path(self, alias: Option<&str>) -> String {
        match self {
            Self::EvictIdle => "/admin/models/evict-idle".to_owned(),
            Self::Load => format!("/admin/models/{}/load", alias.expect("alias")),
            Self::Preload => format!("/admin/models/{}/preload", alias.expect("alias")),
            Self::Unload => format!("/admin/models/{}/unload", alias.expect("alias")),
            Self::ClearQuarantine => {
                format!("/admin/models/{}/clear-quarantine", alias.expect("alias"))
            }
        }
    }
}

#[derive(Debug)]
struct Endpoint {
    host: String,
    port: u16,
}

pub fn run(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    let action_name = arguments.next().ok_or_else(|| {
        "usage: sllm models <load|preload|unload|clear-quarantine|evict-idle> [ALIAS]".to_owned()
    })?;
    let action = Action::parse(&action_name)?;
    let mut alias = None;
    let mut server = DEFAULT_SERVER.to_owned();
    let mut api_key_env = None;
    let mut api_key_file = None;

    let mut rest = arguments.peekable();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--server" => {
                let value = rest
                    .next()
                    .ok_or_else(|| "--server requires a URL".to_owned())?;
                server = value;
            }
            "--api-key-env" => {
                let value = rest.next().ok_or_else(|| {
                    "--api-key-env requires an environment variable name".to_owned()
                })?;
                api_key_env = Some(value);
            }
            "--api-key-file" => {
                let value = rest
                    .next()
                    .ok_or_else(|| "--api-key-file requires a path".to_owned())?;
                api_key_file = Some(value);
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown models option `{value}`"));
            }
            value => {
                if alias.replace(value.to_owned()).is_some() {
                    return Err("only one model alias may be provided".to_owned());
                }
            }
        }
    }

    if api_key_env.is_some() && api_key_file.is_some() {
        return Err("--api-key-env and --api-key-file are mutually exclusive".to_owned());
    }
    if action == Action::EvictIdle {
        if alias.is_some() {
            return Err("evict-idle does not accept a model alias".to_owned());
        }
    } else {
        let value = alias
            .as_deref()
            .ok_or_else(|| format!("models {action_name} requires a model alias"))?;
        validate_alias(value)?;
    }
    let endpoint = parse_endpoint(&server)?;
    let token = if let Some(name) = api_key_env {
        read_env_token(&name)?
    } else if let Some(path) = api_key_file {
        read_file_token(Path::new(&path))?
    } else {
        None
    };

    let response = request(&endpoint, action.path(alias.as_deref()), token.as_deref())?;
    if !response.is_empty() {
        print!("{response}");
        if !response.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

fn validate_alias(alias: &str) -> Result<(), String> {
    if alias.is_empty()
        || alias.len() > MAX_ALIAS_BYTES
        || !alias
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err("model alias must be 1..128 ASCII letters, digits, '.', '_' or '-'".to_owned());
    }
    Ok(())
}

fn parse_endpoint(url: &str) -> Result<Endpoint, String> {
    if url.len() > MAX_URL_BYTES {
        return Err("server URL is too long".to_owned());
    }
    let authority = url
        .strip_prefix("http://")
        .ok_or_else(|| "server URL must use clear-text http://".to_owned())?;
    if authority.is_empty()
        || authority.contains('/')
        || authority.contains('?')
        || authority.contains('#')
        || authority.contains('@')
    {
        return Err("server URL must contain only a loopback host and optional port".to_owned());
    }
    let (host, port) = if authority.starts_with('[') {
        let close = authority
            .find(']')
            .ok_or_else(|| "invalid bracketed loopback host".to_owned())?;
        let host = &authority[1..close];
        if host != "::1" {
            return Err("server host must be localhost, 127.0.0.1, or [::1]".to_owned());
        }
        let port = match authority.as_bytes().get(close + 1) {
            None => 80,
            Some(b':') => parse_port(&authority[close + 2..])?,
            _ => return Err("invalid bracketed loopback host".to_owned()),
        };
        (host.to_owned(), port)
    } else {
        let (host, port) = authority
            .rsplit_once(':')
            .map(|(host, port)| (host, parse_port(port)))
            .unwrap_or((authority, Ok(80)));
        let port = port?;
        if !matches!(host, "localhost" | "127.0.0.1") {
            return Err("server host must be localhost, 127.0.0.1, or [::1]".to_owned());
        }
        (host.to_owned(), port)
    };
    // Resolve localhost before connecting, and fail closed if a resolver maps it
    // to a non-loopback address.
    let addresses = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|_| "server host could not be resolved".to_owned())?
        .collect::<Vec<_>>();
    if addresses.is_empty() || addresses.iter().any(|address| !address.ip().is_loopback()) {
        return Err("server endpoint must resolve only to loopback addresses".to_owned());
    }
    Ok(Endpoint { host, port })
}

fn parse_port(value: &str) -> Result<u16, String> {
    let port = value
        .parse::<u16>()
        .map_err(|_| "server URL port must be 1..65535".to_owned())?;
    if port == 0 {
        return Err("server URL port must be 1..65535".to_owned());
    }
    Ok(port)
}

fn read_env_token(name: &str) -> Result<Option<String>, String> {
    if name.is_empty() || name.len() > 256 || !name.bytes().all(|byte| byte != 0 && byte != b'=') {
        return Err("API key environment variable name is invalid".to_owned());
    }
    let value =
        env::var(name).map_err(|_| format!("API key environment variable {name} is absent"))?;
    validate_token(value).map(Some)
}

fn read_file_token(path: &Path) -> Result<Option<String>, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|_| "API key file could not be inspected".to_owned())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("API key file must be a regular, non-symlink file".to_owned());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let file: File = options
        .open(path)
        .map_err(|_| "API key file could not be opened".to_owned())?;
    let opened = file
        .metadata()
        .map_err(|_| "API key file could not be inspected".to_owned())?;
    if !opened.is_file() || opened.len() > MAX_TOKEN_BYTES as u64 {
        return Err("API key file is missing, not regular, or too large".to_owned());
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    file.take((MAX_TOKEN_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "API key file could not be read".to_owned())?;
    if bytes.len() > MAX_TOKEN_BYTES {
        return Err("API key file is too large".to_owned());
    }
    let value =
        String::from_utf8(bytes).map_err(|_| "API key file must contain UTF-8 text".to_owned())?;
    validate_token(value.trim().to_owned()).map(Some)
}

fn validate_token(value: String) -> Result<String, String> {
    let trimmed = value.trim().to_owned();
    if trimmed.is_empty()
        || trimmed.len() > MAX_TOKEN_BYTES
        || trimmed.chars().any(char::is_control)
    {
        return Err(
            "API key must be non-empty, bounded, and contain no control characters".to_owned(),
        );
    }
    Ok(trimmed)
}

fn request(endpoint: &Endpoint, path: String, token: Option<&str>) -> Result<String, String> {
    let addresses = (endpoint.host.as_str(), endpoint.port)
        .to_socket_addrs()
        .map_err(|_| "server host could not be resolved".to_owned())?
        .filter(|address| address.ip().is_loopback())
        .collect::<Vec<SocketAddr>>();
    let mut stream = addresses
        .iter()
        .find_map(|address| TcpStream::connect_timeout(address, Duration::from_secs(5)).ok())
        .ok_or_else(|| "could not connect to loopback server".to_owned())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|_| "could not configure server read timeout".to_owned())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|_| "could not configure server write timeout".to_owned())?;
    let host_header = if endpoint.host == "::1" {
        format!("[{}]:{}", endpoint.host, endpoint.port)
    } else {
        format!("{}:{}", endpoint.host, endpoint.port)
    };
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\nConnection: close\r\nContent-Length: 0\r\n"
    );
    if let Some(token) = token {
        request.push_str("Authorization: Bearer ");
        request.push_str(token);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|_| "could not send model administration request".to_owned())?;
    let mut bytes = Vec::new();
    stream
        .take((MAX_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "could not read model administration response".to_owned())?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err("model administration response is too large".to_owned());
    }
    parse_response(&bytes)
}

fn parse_response(bytes: &[u8]) -> Result<String, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "model administration response was not valid UTF-8".to_owned())?;
    let (header, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "invalid HTTP response from model administration endpoint".to_owned())?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| "invalid HTTP status from model administration endpoint".to_owned())?;
    if !(200..300).contains(&status) {
        return Err(format!(
            "model administration request failed with HTTP status {status}"
        ));
    }
    Ok(body.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_accepts_loopback_only() {
        assert_eq!(parse_endpoint("http://127.0.0.1:8080").unwrap().port, 8080);
        assert_eq!(parse_endpoint("http://[::1]:8080").unwrap().host, "::1");
        assert_eq!(parse_endpoint("http://[::1]").unwrap().port, 80);
        assert!(parse_endpoint("https://127.0.0.1:8080").is_err());
        assert!(parse_endpoint("http://192.0.2.1:8080").is_err());
        assert!(parse_endpoint("http://127.0.0.1:0").is_err());
        assert!(parse_endpoint("http://127.0.0.1:8080/path").is_err());
    }

    #[test]
    fn aliases_are_bounded_and_path_safe() {
        assert!(validate_alias("qwen-4b.v1").is_ok());
        assert!(validate_alias("a".repeat(MAX_ALIAS_BYTES).as_str()).is_ok());
        assert!(validate_alias(&"a".repeat(MAX_ALIAS_BYTES + 1)).is_err());
        assert!(validate_alias("../escape").is_err());
        assert!(validate_alias("alias/other").is_err());
    }

    #[test]
    fn response_status_is_checked_without_returning_error_body() {
        assert_eq!(
            parse_response(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n").unwrap(),
            ""
        );
        let error = parse_response(b"HTTP/1.1 500 Secret\r\n\r\nprivate details").unwrap_err();
        assert_eq!(
            error,
            "model administration request failed with HTTP status 500"
        );
    }

    #[test]
    fn actions_render_alias_only_paths() {
        assert_eq!(Action::Load.path(Some("model")), "/admin/models/model/load");
        assert_eq!(Action::EvictIdle.path(None), "/admin/models/evict-idle");
    }
}
