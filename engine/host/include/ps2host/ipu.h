/*
 * The IPU, the EE's MPEG decoder (EE User's Manual, chapter 8): what a game's
 * libmpeg drives to play movies. It takes an MPEG-1/2 bitstream through its
 * input FIFO (DMA channel 4, TO_IPU) and decodes what its commands ask -
 * a header field (FDEC), a VLC symbol (VDEC), a macroblock's coefficients
 * through the inverse quantiser and IDCT (BDEC) - and turns macroblocks of
 * YCbCr into RGB (CSC), the results leaving through its output FIFO (DMA
 * channel 3, FROM_IPU). The motion compensation is the EE's, in libmpeg.
 */

#ifndef PS2HOST_IPU_H
#define PS2HOST_IPU_H

namespace ps2host {

/* Hooks the IPU's registers (0x10002000-0x1000203F), its FIFOs (0x10007000,
   0x10007010) and DMA channels 3 and 4 (devices.cpp leaves those two to it). */
void InstallIpu();

}  // namespace ps2host

#endif  /* PS2HOST_IPU_H */
