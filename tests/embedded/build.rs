use std::env;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    // std::fs::write(out.join("memory.x"), include_bytes!("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg-bins=--nmagic");
    //println!("cargo:rustc-link-arg-bins=-Tmemory.x");
    //println!("cargo:rerun-if-changed=memory.x");

    println!("cargo:rustc-link-arg-bins=-Tlink_ram.x");
    println!("cargo:rustc-link-arg-bins=-Tdevice.x");
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");
    println!("cargo:rustc-link-arg-bins=-Tteleprobe.x");

    Ok(())
}
