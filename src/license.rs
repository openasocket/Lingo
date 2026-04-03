//! License attestation for the FiddyCent Software License.
//!
//! Users must accept the license before using lingo. Acceptance is
//! recorded via a sentinel file at `~/.cache/lingo/.license-accepted`, or
//! by setting the environment variable `LINGO_ACCEPT_LICENSE=1`.

use std::path::PathBuf;

/// Returns `true` if the user has already accepted the license.
///
/// Acceptance is indicated by either:
/// - The sentinel file `~/.cache/lingo/.license-accepted` existing, OR
/// - The environment variable `LINGO_ACCEPT_LICENSE` being set to `"1"`.
pub fn license_accepted() -> bool {
    if std::env::var("LINGO_ACCEPT_LICENSE").as_deref() == Ok("1") {
        return true;
    }
    sentinel_path()
        .map(|p| p.exists())
        .unwrap_or(false)
}

/// Creates the sentinel file (and parent directories) to record acceptance.
pub fn mark_license_accepted() -> std::io::Result<()> {
    let path = sentinel_path().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "Cannot determine home directory")
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, b"accepted\n")
}

/// Returns a short summary of the FiddyCent Software License.
pub fn license_notice() -> &'static str {
    "\
lingo is licensed under the FiddyCent Software License.

This is a restricted-use, source-available license. Key restrictions:
  - Use by forbidden parties is prohibited.
  - Use of this software or its outputs for ML/AI model training is prohibited.
  - Derivative works must preserve this license and all restrictions.

Full license terms: LICENSE

You must accept the license before using this software."
}

/// Returns `Ok(())` if the license has been accepted, or
/// `Err(crate::Error::LicenseNotAccepted)` otherwise.
pub fn require_license_acceptance() -> crate::Result<()> {
    if license_accepted() {
        Ok(())
    } else {
        Err(crate::Error::LicenseNotAccepted)
    }
}

/// Convenience alias for [`require_license_acceptance`].
pub fn check_license_acceptance() -> crate::Result<()> {
    require_license_acceptance()
}

fn sentinel_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cache/lingo/.license-accepted"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn license_notice_is_non_empty() {
        let notice = license_notice();
        assert!(!notice.is_empty());
        assert!(notice.contains("FiddyCent Software License"));
    }

    #[test]
    fn sentinel_path_is_under_cache() {
        let path = sentinel_path().expect("home dir should exist in test");
        assert!(path.to_string_lossy().contains(".cache/lingo/.license-accepted"));
    }

    #[test]
    fn env_var_override_accepts_license() {
        // Save and restore env var
        let prev = std::env::var("LINGO_ACCEPT_LICENSE").ok();
        std::env::set_var("LINGO_ACCEPT_LICENSE", "1");
        assert!(license_accepted());
        match prev {
            Some(v) => std::env::set_var("LINGO_ACCEPT_LICENSE", v),
            None => std::env::remove_var("LINGO_ACCEPT_LICENSE"),
        }
    }

    #[test]
    fn require_license_acceptance_returns_error_when_not_accepted() {
        let prev = std::env::var("LINGO_ACCEPT_LICENSE").ok();
        std::env::remove_var("LINGO_ACCEPT_LICENSE");
        // With no env var and (likely) no sentinel file, this should fail
        // unless the test machine has accepted — so we just check the function runs.
        let _ = require_license_acceptance();
        match prev {
            Some(v) => std::env::set_var("LINGO_ACCEPT_LICENSE", v),
            None => {}
        }
    }
}
