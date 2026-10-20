// Link the folded table into the executable; no runtime file dependency.
.section .rodata
.balign 64
.global architecture_table
architecture_table:
.incbin "architecture.bin"
.section .note.GNU-stack,"",@progbits
