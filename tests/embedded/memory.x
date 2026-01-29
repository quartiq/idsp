/* Simple run-from-ram layout calling 3/4 of the DTCM "FLASH" */
MEMORY
{
  FLASH  (rwx) : ORIGIN = 0x20000000, LENGTH = 96K  /* DTCM */
  RAM    (rwx) : ORIGIN = 0x20018000, LENGTH = 32K  /* DTCM */
}
