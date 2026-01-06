// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(all(feature = "shuttle", test))]
pub(crate) use shuttle::sync::*;

#[cfg(all(feature = "shuttle", test))]
pub(crate) use shuttle::thread;

#[cfg(not(all(feature = "shuttle", test)))]
pub(crate) use std::sync::*;

#[cfg(not(all(feature = "shuttle", test)))]
#[allow(unused_imports)]
pub(crate) use std::thread;
