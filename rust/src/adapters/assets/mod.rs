//! The shared model cache: downloads, verification, locks, and the store over them.

pub mod download;
pub mod file_lock;
pub mod resolver;
pub mod store;

#[cfg(test)]
mod store_tests;
