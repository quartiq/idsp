/* Simple run-from-ram layout "FLASH" aliasing "RAM" */
MEMORY
{
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K  /* DTCM */
}
REGION_ALIAS("FLASH", RAM);
