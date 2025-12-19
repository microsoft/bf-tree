# Debugging tips

First tip: use a debugger, i.e., lldb.

### LLDB tips

##### LLDB cheat sheet

https://github.com/carolanitz/DebuggingFun/blob/master/lldb%20cheat%20sheet.pdf

##### Display format

When watching a variable, we often need to show it in hex, binary, or decimal. This can easily be done by adding `,h`, `,b`, or `,d` after the variable name. For example, `return_addr,h` will display `return_addr` in hex.
More information can be found [here](https://github.com/vadimcn/codelldb/blob/master/MANUAL.md#formatting).


##### Bf-Tree specific formatters

General purpose debuggers don't understand Bf-Tree's bit-packed layout. To make debugging easier, we maintained our own formatters for several types. They should be automatically loaded if you are using the recommended `codelldb` plugin and the default settings in .vscode.

Note that the formatter is in Python, meaning that we actually maintained two formatting implementations: one in Rust and one in Python. This means that whenever we update the Rust's layout, we need to update the formatter in Python as well.
