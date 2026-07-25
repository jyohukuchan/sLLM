/* Root-only launcher for the sealed AQ4 hardening operation payload. */

/* Needed for O_NOFOLLOW, O_CLOEXEC, lstat, strtok_r, and PATH_MAX on glibc. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef AQ4_OPERATION_SCRIPT_PATH
#error "AQ4_OPERATION_SCRIPT_PATH is required"
#endif

#ifndef AQ4_OPERATION_SCRIPT_SHA256
#error "AQ4_OPERATION_SCRIPT_SHA256 is required"
#endif

extern char **environ;

static int safe_ancestry(const char *path) {
    char current[PATH_MAX] = "/";
    char scratch[PATH_MAX];
    struct stat metadata;
    char *part;
    char *save = NULL;

    if (strlen(path) >= sizeof(scratch) || path[0] != '/') {
        return 0;
    }
    strcpy(scratch, path + 1);
    for (part = strtok_r(scratch, "/", &save); part != NULL;
         part = strtok_r(NULL, "/", &save)) {
        size_t used = strlen(current);
        if (used + strlen(part) + 2 > sizeof(current)) {
            return 0;
        }
        if (used > 1) {
            strcat(current, "/");
        }
        strcat(current, part);
        if (lstat(current, &metadata) != 0 || S_ISLNK(metadata.st_mode) ||
            metadata.st_uid != 0 || (metadata.st_mode & S_IWGRP) ||
            (metadata.st_mode & S_IWOTH)) {
            return 0;
        }
    }
    return 1;
}

static int script_digest_matches(void) {
    unsigned char buffer[65536];
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    char encoded[EVP_MAX_MD_SIZE * 2 + 1];
    EVP_MD_CTX *context = NULL;
    struct stat metadata;
    ssize_t read_count;
    int descriptor = -1;
    int ok = 0;

    if (!safe_ancestry(AQ4_OPERATION_SCRIPT_PATH)) {
        return 0;
    }
    descriptor = open(AQ4_OPERATION_SCRIPT_PATH, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0 || fstat(descriptor, &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_uid != 0 ||
        (metadata.st_mode & 0777) != 0444 || metadata.st_nlink != 1 ||
        metadata.st_size < 1 || metadata.st_size > 1048576) {
        goto done;
    }
    context = EVP_MD_CTX_new();
    if (context == NULL || EVP_DigestInit_ex(context, EVP_sha256(), NULL) != 1) {
        goto done;
    }
    while ((read_count = read(descriptor, buffer, sizeof(buffer))) > 0) {
        if (EVP_DigestUpdate(context, buffer, (size_t)read_count) != 1) {
            goto done;
        }
    }
    if (read_count != 0 || EVP_DigestFinal_ex(context, digest, &digest_length) != 1 ||
        digest_length != 32) {
        goto done;
    }
    for (unsigned int index = 0; index < digest_length; ++index) {
        snprintf(encoded + index * 2, 3, "%02x", digest[index]);
    }
    encoded[64] = '\0';
    ok = strcmp(encoded, AQ4_OPERATION_SCRIPT_SHA256) == 0;

done:
    if (context != NULL) {
        EVP_MD_CTX_free(context);
    }
    if (descriptor >= 0) {
        close(descriptor);
    }
    return ok;
}

int main(int argc, char **argv) {
    char **python_argv;

    if (geteuid() != 0 || argc < 2 || !script_digest_matches()) {
        fputs("AQ4 hardening operation launcher rejected its invocation\n", stderr);
        return 126;
    }
    python_argv = calloc((size_t)argc + 5, sizeof(*python_argv));
    if (python_argv == NULL) {
        return 127;
    }
    python_argv[0] = "/usr/bin/python3";
    python_argv[1] = "-I";
    python_argv[2] = "-S";
    python_argv[3] = "-B";
    python_argv[4] = AQ4_OPERATION_SCRIPT_PATH;
    for (int index = 1; index < argc; ++index) {
        python_argv[index + 4] = argv[index];
    }
    execve(python_argv[0], python_argv, environ);
    fprintf(stderr, "AQ4 hardening operation launcher could not start Python: %s\n", strerror(errno));
    free(python_argv);
    return 127;
}
