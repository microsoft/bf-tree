// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(test)]
pub(crate) fn install_value_to_buffer(buffer: &mut [usize], key_id: usize) -> &[u8] {
    for i in buffer.iter_mut() {
        *i = key_id;
    }

    unsafe {
        let ptr = buffer.as_ptr();
        std::slice::from_raw_parts(ptr as *const u8, buffer.len() * 8)
    }
}
