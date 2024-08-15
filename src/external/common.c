#include "common.h"

typedef union {
  float float_part;
  MKL_BF16 int_part[2];
} conv_union_bf16;

typedef union {
  MKL_F16  raw;
  struct {
    unsigned int frac : 10;
    unsigned int exp  :  5;
    unsigned int sign :  1;
  } bits;
} conv_union_f16;

typedef union {
  float raw;
  struct {
    unsigned int frac : 23;
    unsigned int exp  :  8;
    unsigned int sign :  1;
  } bits;
} conv_union_f32;

float b2f(MKL_BF16 src) {
  conv_union_bf16 conv;
  conv.int_part[0] = 0;
  conv.int_part[1] = src;
  return conv.float_part;
}

MKL_BF16 f2b(float src) {
  conv_union_bf16 conv;
  conv.float_part = src;
  return conv.int_part[1];
}

float h2f(MKL_F16 x) { 
  conv_union_f16 src;
  conv_union_f32 dst;

  src.raw  = x;
  dst.raw  = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x01fU) {
    dst.bits.exp  = 0xffU;
    if (src.bits.frac > 0) {
      dst.bits.frac = ((src.bits.frac | 0x200U) << 13);
    }
  } else if (src.bits.exp > 0x00U) {
    dst.bits.exp  = src.bits.exp + ((1 << 7) - (1 << 4));
    dst.bits.frac = (src.bits.frac << 13);
  } else {
    unsigned int v = (src.bits.frac << 13);

    if (v > 0) {
      dst.bits.exp = 0x71;
      while ((v & 0x800000UL) == 0) {
	dst.bits.exp --;
	v <<= 1;
      }
      dst.bits.frac = v;
    }
  }

  return dst.raw;
}

MKL_F16 f2h(float x) { 
  conv_union_f32 src;
  conv_union_f16 dst;

  src.raw  = x;
  dst.raw  = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x0ffU) {
    dst.bits.exp  = 0x01fU;
    dst.bits.frac = (src.bits.frac >> 13);
    if (src.bits.frac > 0) dst.bits.frac |= 0x200U;
  } else if (src.bits.exp >= 0x08fU) {
    dst.bits.exp  = 0x01fU;
    dst.bits.frac = 0x000U;
  } else if (src.bits.exp >= 0x071U){
    dst.bits.exp  = src.bits.exp + ((1 << 4) - (1 << 7));
    dst.bits.frac = (src.bits.frac >> 13);
  } else if (src.bits.exp >= 0x067U){
    dst.bits.exp  = 0x000;
    if (src.bits.frac > 0) {
      dst.bits.frac = (((1U << 23) | src.bits.frac) >> 14);
    } else {
      dst.bits.frac = 1;
    }
  }

  return dst.raw;
}
