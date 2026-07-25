
benchmarks/results/2026-07-26/sq8-r9700-attention-phase1-v0.1/decode/code-objects/shared-fallback-runtime-dump/_code_object0010.o:	file format elf64-amdgpu

Disassembly of section .text:

0000000000002a00 <ullm_paged_decode_attn_f32_kernel>:
	s_clause 0x3                                               // 000000002A00: BF850003
	s_load_b96 s[28:30], s[0:1], 0x68                          // 000000002A04: F400A700 F8000068
	s_load_b256 s[36:43], s[0:1], 0x40                         // 000000002A0C: F4006900 F8000040
	s_load_b512 s[12:27], s[0:1], 0x0                          // 000000002A14: F4008300 F8000000
	s_load_b32 s4, s[0:1], 0x7c                                // 000000002A1C: F4000100 F800007C
	s_mov_b32 s3, 0                                            // 000000002A24: BE830080
	s_wait_kmcnt 0x0                                           // 000000002A28: BFC70000
	s_mov_b32 s2, s30                                          // 000000002A2C: BE82001E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002A30: BF870009
	s_cmp_lg_u64 s[36:37], s[2:3]                              // 000000002A34: BF110224
	s_mov_b32 s2, -1                                           // 000000002A38: BE8200C1
	s_cbranch_scc0 8                                           // 000000002A3C: BFA10008 <ullm_paged_decode_attn_f32_kernel+0x60>
	s_load_b32 s33, s[0:1], 0x60                               // 000000002A40: F4000840 F8000060
	s_mov_b32 s10, ttmp9                                       // 000000002A48: BE8A0075
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000002A4C: 916A027E
	s_cbranch_vccz 23                                          // 000000002A50: BFA30017 <ullm_paged_decode_attn_f32_kernel+0xb0>
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000002A54: 916A037E
	s_cbranch_vccz 1409                                        // 000000002A58: BFA30581 <ullm_paged_decode_attn_f32_kernel+0x1660>
	s_endpgm                                                   // 000000002A5C: BFB00000
	v_cmp_lt_u64_e64 s2, 0x100, s[40:41]                       // 000000002A60: D4590002 000050FF 00000100
	v_cmp_lt_u64_e64 s3, 0x100, s[42:43]                       // 000000002A6C: D4590003 000054FF 00000100
	s_and_b32 s5, 0xffff, s4                                   // 000000002A78: 8B0504FF 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002A80: BF870009
	s_cmp_lg_u32 s5, 0x100                                     // 000000002A84: BF07FF05 00000100
	s_cselect_b32 s5, -1, 0                                    // 000000002A8C: 980580C1
	s_or_b32 s2, s2, s3                                        // 000000002A90: 8C020302
	s_mov_b32 s3, -1                                           // 000000002A94: BE8300C1
	s_or_b32 s2, s2, s5                                        // 000000002A98: 8C020502
	s_load_b32 s33, s[0:1], 0x60                               // 000000002A9C: F4000840 F8000060
	s_mov_b32 s10, ttmp9                                       // 000000002AA4: BE8A0075
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000002AA8: 916A027E
	s_cbranch_vccnz 65513                                      // 000000002AAC: BFA4FFE9 <ullm_paged_decode_attn_f32_kernel+0x54>
	v_mov_b32_e32 v1, 0                                        // 000000002AB0: 7E020280
	s_and_b32 s0, 0xffff, s4                                   // 000000002AB4: 8B0004FF 0000FFFF
	s_mov_b32 s11, exec_lo                                     // 000000002ABC: BE8B007E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002AC0: BF870001
	v_mad_co_u64_u32 v[2:3], null, s0, s10, v[0:1]             // 000000002AC4: D6FE7C02 04001400
	s_mul_u64 s[0:1], s[42:43], s[36:37]                       // 000000002ACC: AA80242A
	s_wait_alu 0xfffe                                          // 000000002AD0: BF88FFFE
	v_cmpx_gt_u64_e64 s[0:1], v[2:3]                           // 000000002AD4: D4DC007E 00020400
	s_cbranch_execz 1374                                       // 000000002ADC: BFA5055E <ullm_paged_decode_attn_f32_kernel+0x1658>
	v_or_b32_e32 v5, s43, v3                                   // 000000002AE0: 380A062B
	v_mov_b32_e32 v4, v1                                       // 000000002AE4: 7E080301
	s_mov_b32 s0, exec_lo                                      // 000000002AE8: BE80007E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002AEC: BF870001
	v_cmpx_ne_u64_e32 0, v[4:5]                                // 000000002AF0: 7DBA0880
	s_wait_alu 0xfffe                                          // 000000002AF4: BF88FFFE
	s_xor_b32 s1, exec_lo, s0                                  // 000000002AF8: 8D01007E
	s_cbranch_execz 163                                        // 000000002AFC: BFA500A3 <ullm_paged_decode_attn_f32_kernel+0x38c>
	s_cvt_f32_u32 s0, s42                                      // 000000002B00: BE80652A
	s_cvt_f32_u32 s2, s43                                      // 000000002B04: BE82652B
	s_sub_nc_u64 s[4:5], 0, s[42:43]                           // 000000002B08: AA042A80
	s_mov_b32 s9, 0                                            // 000000002B0C: BE890080
	s_wait_alu 0xfffe                                          // 000000002B10: BF88FFFE
	s_fmamk_f32 s0, s2, 0x4f800000, s0                         // 000000002B14: A3000002 4F800000
	s_wait_alu 0xfffe                                          // 000000002B1C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000002B20: BF87029A
	v_s_rcp_f32 s0, s0                                         // 000000002B24: D6840000 00000000
	s_mul_f32 s0, s0, 0x5f7ffffc                               // 000000002B2C: A200FF00 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000002B34: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000002B38: BF87059A
	s_mul_f32 s2, s0, 0x2f800000                               // 000000002B3C: A202FF00 2F800000
	s_trunc_f32 s2, s2                                         // 000000002B44: BE826202
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002B48: BF8704BB
	s_fmamk_f32 s0, s2, 0xcf800000, s0                         // 000000002B4C: A3000002 CF800000
	s_cvt_u32_f32 s3, s2                                       // 000000002B54: BE836702
	s_wait_alu 0xfffe                                          // 000000002B58: BF88FFFE
	s_cvt_u32_f32 s2, s0                                       // 000000002B5C: BE826700
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002B60: BF87049B
	s_mul_u64 s[6:7], s[4:5], s[2:3]                           // 000000002B64: AA860204
	s_mul_hi_u32 s31, s2, s7                                   // 000000002B68: 969F0702
	s_mul_i32 s30, s2, s7                                      // 000000002B6C: 961E0702
	s_mul_hi_u32 s8, s2, s6                                    // 000000002B70: 96880602
	s_mul_i32 s34, s3, s6                                      // 000000002B74: 96220603
	s_add_nc_u64 s[30:31], s[8:9], s[30:31]                    // 000000002B78: A99E1E08
	s_mul_hi_u32 s0, s3, s6                                    // 000000002B7C: 96800603
	s_mul_hi_u32 s35, s3, s7                                   // 000000002B80: 96A30703
	s_mul_i32 s6, s3, s7                                       // 000000002B84: 96060703
	s_add_co_u32 s7, s30, s34                                  // 000000002B88: 8007221E
	s_wait_alu 0xfffe                                          // 000000002B8C: BF88FFFE
	s_add_co_ci_u32 s8, s31, s0                                // 000000002B90: 8208001F
	s_add_co_ci_u32 s7, s35, 0                                 // 000000002B94: 82078023
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002B98: BF870499
	s_add_nc_u64 s[6:7], s[8:9], s[6:7]                        // 000000002B9C: A9860608
	s_add_co_u32 s2, s2, s6                                    // 000000002BA0: 80020602
	s_cselect_b32 s0, -1, 0                                    // 000000002BA4: 980080C1
	s_wait_alu 0xfffe                                          // 000000002BA8: BF88FFFE
	s_cmp_lg_u32 s0, 0                                         // 000000002BAC: BF078000
	s_add_co_ci_u32 s3, s3, s7                                 // 000000002BB0: 82030703
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002BB4: BF870499
	s_mul_u64 s[4:5], s[4:5], s[2:3]                           // 000000002BB8: AA840204
	s_mul_hi_u32 s7, s2, s5                                    // 000000002BBC: 96870502
	s_mul_i32 s6, s2, s5                                       // 000000002BC0: 96060502
	s_mul_hi_u32 s8, s2, s4                                    // 000000002BC4: 96880402
	s_mul_i32 s30, s3, s4                                      // 000000002BC8: 961E0403
	s_add_nc_u64 s[6:7], s[8:9], s[6:7]                        // 000000002BCC: A9860608
	s_mul_hi_u32 s0, s3, s4                                    // 000000002BD0: 96800403
	s_mul_hi_u32 s31, s3, s5                                   // 000000002BD4: 969F0503
	s_mul_i32 s4, s3, s5                                       // 000000002BD8: 96040503
	s_add_co_u32 s5, s6, s30                                   // 000000002BDC: 80051E06
	s_wait_alu 0xfffe                                          // 000000002BE0: BF88FFFE
	s_add_co_ci_u32 s8, s7, s0                                 // 000000002BE4: 82080007
	s_add_co_ci_u32 s5, s31, 0                                 // 000000002BE8: 8205801F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002BEC: BF870499
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]                        // 000000002BF0: A9840408
	s_add_co_u32 s0, s2, s4                                    // 000000002BF4: 80000402
	s_cselect_b32 s2, -1, 0                                    // 000000002BF8: 980280C1
	s_wait_alu 0xfffe                                          // 000000002BFC: BF88FFFE
	v_mul_hi_u32 v1, v2, s0                                    // 000000002C00: D72D0001 00000102
	s_cmp_lg_u32 s2, 0                                         // 000000002C08: BF078002
	v_mad_co_u64_u32 v[6:7], null, v3, s0, 0                   // 000000002C0C: D6FE7C06 02000103
	s_add_co_ci_u32 s2, s3, s5                                 // 000000002C14: 82020503
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002C18: BF870129
	v_mad_co_u64_u32 v[4:5], null, v2, s2, 0                   // 000000002C1C: D6FE7C04 02000502
	v_mad_co_u64_u32 v[8:9], null, v3, s2, 0                   // 000000002C24: D6FE7C08 02000503
	v_add_co_u32 v1, vcc_lo, v1, v4                            // 000000002C2C: D7006A01 00020901
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002C34: BF870111
	v_add_co_ci_u32_e64 v4, null, 0, v5, vcc_lo                // 000000002C38: D5207C04 01AA0A80
	v_add_co_u32 v1, vcc_lo, v1, v6                            // 000000002C40: D7006A01 00020D01
	s_wait_alu 0xfffd                                          // 000000002C48: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000002C4C: BF870132
	v_add_co_ci_u32_e32 v1, vcc_lo, v4, v7, vcc_lo             // 000000002C50: 40020F04
	s_wait_alu 0xfffd                                          // 000000002C54: BF88FFFD
	v_add_co_ci_u32_e32 v4, vcc_lo, 0, v9, vcc_lo              // 000000002C58: 40081280
	v_add_co_u32 v1, vcc_lo, v1, v8                            // 000000002C5C: D7006A01 00021101
	s_wait_alu 0xfffd                                          // 000000002C64: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002C68: BF870112
	v_add_co_ci_u32_e64 v6, null, 0, v4, vcc_lo                // 000000002C6C: D5207C06 01AA0880
	v_mul_lo_u32 v7, s43, v1                                   // 000000002C74: D72C0007 0002022B
	v_mad_co_u64_u32 v[4:5], null, s42, v1, 0                  // 000000002C7C: D6FE7C04 0202022A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002C84: BF870113
	v_mul_lo_u32 v8, s42, v6                                   // 000000002C88: D72C0008 00020C2A
	v_sub_co_u32 v4, vcc_lo, v2, v4                            // 000000002C90: D7016A04 00020902
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000002C98: BF8701C2
	v_add3_u32 v5, v5, v8, v7                                  // 000000002C9C: D6550005 041E1105
	v_add_co_u32 v8, s0, v1, 2                                 // 000000002CA4: D7000008 00010501
	s_wait_alu 0xf1ff                                          // 000000002CAC: BF88F1FF
	v_add_co_ci_u32_e64 v9, null, 0, v6, s0                    // 000000002CB0: D5207C09 00020C80
	v_sub_nc_u32_e32 v7, v3, v5                                // 000000002CB8: 4C0E0B03
	v_sub_co_u32 v10, s0, v4, s42                              // 000000002CBC: D701000A 00005504
	s_wait_alu 0xfffd                                          // 000000002CC4: BF88FFFD
	v_sub_co_ci_u32_e64 v5, null, v3, v5, vcc_lo               // 000000002CC8: D5217C05 01AA0B03
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002CD0: BF870193
	v_subrev_co_ci_u32_e64 v7, null, s43, v7, vcc_lo           // 000000002CD4: D5227C07 01AA0E2B
	v_cmp_le_u32_e32 vcc_lo, s42, v10                          // 000000002CDC: 7C96142A
	s_wait_alu 0xf1ff                                          // 000000002CE0: BF88F1FF
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000002CE4: BF8701C2
	v_subrev_co_ci_u32_e64 v7, null, 0, v7, s0                 // 000000002CE8: D5227C07 00020E80
	s_wait_alu 0xfffd                                          // 000000002CF0: BF88FFFD
	v_cndmask_b32_e64 v10, 0, -1, vcc_lo                       // 000000002CF4: D501000A 01A98280
	v_cmp_eq_u32_e64 s0, s43, v5                               // 000000002CFC: D44A0000 00020A2B
	v_cmp_le_u32_e32 vcc_lo, s43, v7                           // 000000002D04: 7C960E2B
	s_wait_alu 0xfffd                                          // 000000002D08: BF88FFFD
	v_cndmask_b32_e64 v11, 0, -1, vcc_lo                       // 000000002D0C: D501000B 01A98280
	v_cmp_le_u32_e32 vcc_lo, s42, v4                           // 000000002D14: 7C96082A
	s_wait_alu 0xfffd                                          // 000000002D18: BF88FFFD
	v_cndmask_b32_e64 v4, 0, -1, vcc_lo                        // 000000002D1C: D5010004 01A98280
	v_cmp_le_u32_e32 vcc_lo, s43, v5                           // 000000002D24: 7C960A2B
	s_wait_alu 0xfffd                                          // 000000002D28: BF88FFFD
	v_cndmask_b32_e64 v12, 0, -1, vcc_lo                       // 000000002D2C: D501000C 01A98280
	v_cmp_eq_u32_e32 vcc_lo, s43, v7                           // 000000002D34: 7C940E2B
	s_wait_alu 0xf1ff                                          // 000000002D38: BF88F1FF
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002D3C: BF870002
	v_cndmask_b32_e64 v4, v12, v4, s0                          // 000000002D40: D5010004 0002090C
	s_wait_alu 0xfffd                                          // 000000002D48: BF88FFFD
	v_cndmask_b32_e32 v7, v11, v10, vcc_lo                     // 000000002D4C: 020E150B
	v_add_co_u32 v10, vcc_lo, v1, 1                            // 000000002D50: D7006A0A 00010301
	s_wait_alu 0xfffd                                          // 000000002D58: BF88FFFD
	v_add_co_ci_u32_e64 v11, null, 0, v6, vcc_lo               // 000000002D5C: D5207C0B 01AA0C80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002D64: BF870123
	v_cmp_ne_u32_e32 vcc_lo, 0, v7                             // 000000002D68: 7C9A0E80
	s_wait_alu 0xfffd                                          // 000000002D6C: BF88FFFD
	v_dual_cndmask_b32 v8, v10, v8 :: v_dual_cndmask_b32 v5, v11, v9// 000000002D70: CA52110A 0804130B
	v_cmp_ne_u32_e32 vcc_lo, 0, v4                             // 000000002D78: 7C9A0880
	s_wait_alu 0xfffd                                          // 000000002D7C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002D80: BF870002
	v_dual_cndmask_b32 v7, v6, v5 :: v_dual_cndmask_b32 v6, v1, v8// 000000002D84: CA520B06 07061101
	s_wait_alu 0xfffe                                          // 000000002D8C: BF88FFFE
	s_and_not1_saveexec_b32 s0, s1                             // 000000002D90: BE803001
	s_cbranch_execz 35                                         // 000000002D94: BFA50023 <ullm_paged_decode_attn_f32_kernel+0x424>
	v_cvt_f32_u32_e32 v1, s42                                  // 000000002D98: 7E020C2A
	s_sub_co_i32 s1, 0, s42                                    // 000000002D9C: 81812A80
	v_mov_b32_e32 v7, 0                                        // 000000002DA0: 7E0E0280
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000002DA4: BF870292
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000002DA8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000002DAC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002DB4: BF8700A1
	v_cvt_u32_f32_e32 v1, v1                                   // 000000002DB8: 7E020F01
	s_wait_alu 0xfffe                                          // 000000002DBC: BF88FFFE
	v_mul_lo_u32 v4, s1, v1                                    // 000000002DC0: D72C0004 00020201
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002DC8: BF870091
	v_mul_hi_u32 v4, v1, v4                                    // 000000002DCC: D72D0004 00020901
	v_add_nc_u32_e32 v1, v1, v4                                // 000000002DD4: 4A020901
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002DD8: BF870091
	v_mul_hi_u32 v1, v2, v1                                    // 000000002DDC: D72D0001 00020302
	v_mul_lo_u32 v4, v1, s42                                   // 000000002DE4: D72C0004 00005501
	v_add_nc_u32_e32 v5, 1, v1                                 // 000000002DEC: 4A0A0281
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002DF0: BF870092
	v_sub_nc_u32_e32 v4, v2, v4                                // 000000002DF4: 4C080902
	v_subrev_nc_u32_e32 v6, s42, v4                            // 000000002DF8: 4E0C082A
	v_cmp_le_u32_e32 vcc_lo, s42, v4                           // 000000002DFC: 7C96082A
	s_wait_alu 0xfffd                                          // 000000002E00: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002E04: BF870092
	v_dual_cndmask_b32 v4, v4, v6 :: v_dual_cndmask_b32 v1, v1, v5// 000000002E08: CA520D04 04000B01
	v_cmp_le_u32_e32 vcc_lo, s42, v4                           // 000000002E10: 7C96082A
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002E14: BF8700A2
	v_add_nc_u32_e32 v5, 1, v1                                 // 000000002E18: 4A0A0281
	s_wait_alu 0xfffd                                          // 000000002E1C: BF88FFFD
	v_cndmask_b32_e32 v6, v1, v5, vcc_lo                       // 000000002E20: 020C0B01
	s_wait_alu 0xfffe                                          // 000000002E24: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002E28: 8C7E007E
	s_or_b64 s[0:1], s[36:37], s[38:39]                        // 000000002E2C: 8C802624
	s_mov_b32 s0, 0                                            // 000000002E30: BE800080
	s_wait_alu 0xfffe                                          // 000000002E34: BF88FFFE
	s_cmp_lg_u64 s[0:1], 0                                     // 000000002E38: BF118000
	s_cbranch_scc0 679                                         // 000000002E3C: BFA102A7 <ullm_paged_decode_attn_f32_kernel+0xedc>
	s_cvt_f32_u32 s1, s38                                      // 000000002E40: BE816526
	s_cvt_f32_u32 s2, s39                                      // 000000002E44: BE826527
	s_sub_nc_u64 s[4:5], 0, s[38:39]                           // 000000002E48: AA042680
	s_mov_b32 s7, s0                                           // 000000002E4C: BE870000
	s_mov_b32 s31, s0                                          // 000000002E50: BE9F0000
	s_wait_alu 0xfffe                                          // 000000002E54: BF88FFFE
	s_fmamk_f32 s1, s2, 0x4f800000, s1                         // 000000002E58: A3010102 4F800000
	s_wait_alu 0xfffe                                          // 000000002E60: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000002E64: BF87029A
	v_s_rcp_f32 s1, s1                                         // 000000002E68: D6840001 00000001
	s_mul_f32 s1, s1, 0x5f7ffffc                               // 000000002E70: A201FF01 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000002E78: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000002E7C: BF87052A
	s_mul_f32 s2, s1, 0x2f800000                               // 000000002E80: A202FF01 2F800000
	s_wait_alu 0xfffe                                          // 000000002E88: BF88FFFE
	s_trunc_f32 s2, s2                                         // 000000002E8C: BE826202
	s_wait_alu 0xfffe                                          // 000000002E90: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002E94: BF8704BA
	s_fmamk_f32 s1, s2, 0xcf800000, s1                         // 000000002E98: A3010102 CF800000
	s_cvt_u32_f32 s3, s2                                       // 000000002EA0: BE836702
	s_wait_alu 0xfffe                                          // 000000002EA4: BF88FFFE
	s_cvt_u32_f32 s2, s1                                       // 000000002EA8: BE826701
	s_wait_alu 0xfffe                                          // 000000002EAC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002EB0: BF87049A
	s_mul_u64 s[8:9], s[4:5], s[2:3]                           // 000000002EB4: AA880204
	s_mul_hi_u32 s35, s2, s9                                   // 000000002EB8: 96A30902
	s_mul_i32 s34, s2, s9                                      // 000000002EBC: 96220902
	s_mul_hi_u32 s6, s2, s8                                    // 000000002EC0: 96860802
	s_mul_i32 s30, s3, s8                                      // 000000002EC4: 961E0803
	s_add_nc_u64 s[6:7], s[6:7], s[34:35]                      // 000000002EC8: A9862206
	s_mul_hi_u32 s1, s3, s8                                    // 000000002ECC: 96810803
	s_mul_hi_u32 s44, s3, s9                                   // 000000002ED0: 96AC0903
	s_add_co_u32 s6, s6, s30                                   // 000000002ED4: 80061E06
	s_wait_alu 0xfffe                                          // 000000002ED8: BF88FFFE
	s_add_co_ci_u32 s30, s7, s1                                // 000000002EDC: 821E0107
	s_mul_i32 s8, s3, s9                                       // 000000002EE0: 96080903
	s_add_co_ci_u32 s9, s44, 0                                 // 000000002EE4: 8209802C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002EE8: BF870009
	s_add_nc_u64 s[6:7], s[30:31], s[8:9]                      // 000000002EEC: A986081E
	s_mov_b32 s9, s0                                           // 000000002EF0: BE890000
	s_add_co_u32 s2, s2, s6                                    // 000000002EF4: 80020602
	s_cselect_b32 s1, -1, 0                                    // 000000002EF8: 980180C1
	s_wait_alu 0xfffe                                          // 000000002EFC: BF88FFFE
	s_cmp_lg_u32 s1, 0                                         // 000000002F00: BF078001
	s_add_co_ci_u32 s3, s3, s7                                 // 000000002F04: 82030703
	s_mov_b32 s7, s0                                           // 000000002F08: BE870000
	s_wait_alu 0xfffe                                          // 000000002F0C: BF88FFFE
	s_mul_u64 s[4:5], s[4:5], s[2:3]                           // 000000002F10: AA840204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002F14: BF870009
	s_mul_hi_u32 s31, s2, s5                                   // 000000002F18: 969F0502
	s_mul_i32 s30, s2, s5                                      // 000000002F1C: 961E0502
	s_mul_hi_u32 s6, s2, s4                                    // 000000002F20: 96860402
	s_mul_i32 s8, s3, s4                                       // 000000002F24: 96080403
	s_add_nc_u64 s[6:7], s[6:7], s[30:31]                      // 000000002F28: A9861E06
	s_mul_hi_u32 s1, s3, s4                                    // 000000002F2C: 96810403
	s_mul_hi_u32 s34, s3, s5                                   // 000000002F30: 96A20503
	s_mul_i32 s4, s3, s5                                       // 000000002F34: 96040503
	s_add_co_u32 s5, s6, s8                                    // 000000002F38: 80050806
	s_wait_alu 0xfffe                                          // 000000002F3C: BF88FFFE
	s_add_co_ci_u32 s8, s7, s1                                 // 000000002F40: 82080107
	s_add_co_ci_u32 s5, s34, 0                                 // 000000002F44: 82058022
	s_mov_b32 s7, s0                                           // 000000002F48: BE870000
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]                        // 000000002F4C: A9840408
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002F50: BF870009
	s_add_co_u32 s1, s2, s4                                    // 000000002F54: 80010402
	s_cselect_b32 s2, -1, 0                                    // 000000002F58: 980280C1
	s_wait_alu 0xfffe                                          // 000000002F5C: BF88FFFE
	s_mul_hi_u32 s6, s36, s1                                   // 000000002F60: 96860124
	s_cmp_lg_u32 s2, 0                                         // 000000002F64: BF078002
	s_mul_hi_u32 s8, s37, s1                                   // 000000002F68: 96880125
	s_add_co_ci_u32 s4, s3, s5                                 // 000000002F6C: 82040503
	s_mul_i32 s1, s37, s1                                      // 000000002F70: 96010125
	s_mul_hi_u32 s3, s36, s4                                   // 000000002F74: 96830424
	s_mul_i32 s2, s36, s4                                      // 000000002F78: 96020424
	s_mul_hi_u32 s5, s37, s4                                   // 000000002F7C: 96850425
	s_wait_alu 0xfffe                                          // 000000002F80: BF88FFFE
	s_add_nc_u64 s[2:3], s[6:7], s[2:3]                        // 000000002F84: A9820206
	s_mul_i32 s4, s37, s4                                      // 000000002F88: 96040425
	s_wait_alu 0xfffe                                          // 000000002F8C: BF88FFFE
	s_add_co_u32 s1, s2, s1                                    // 000000002F90: 80010102
	s_add_co_ci_u32 s8, s3, s8                                 // 000000002F94: 82080803
	s_add_co_ci_u32 s5, s5, 0                                  // 000000002F98: 82058005
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002F9C: BF8704B9
	s_add_nc_u64 s[2:3], s[8:9], s[4:5]                        // 000000002FA0: A9820408
	s_wait_alu 0xfffe                                          // 000000002FA4: BF88FFFE
	s_mul_u64 s[4:5], s[38:39], s[2:3]                         // 000000002FA8: AA840226
	s_sub_co_u32 s1, s36, s4                                   // 000000002FAC: 80810424
	s_cselect_b32 s4, -1, 0                                    // 000000002FB0: 980480C1
	s_sub_co_i32 s6, s37, s5                                   // 000000002FB4: 81860525
	s_cmp_lg_u32 s4, 0                                         // 000000002FB8: BF078004
	s_sub_co_ci_u32 s6, s6, s39                                // 000000002FBC: 82862706
	s_wait_alu 0xfffe                                          // 000000002FC0: BF88FFFE
	s_sub_co_u32 s7, s1, s38                                   // 000000002FC4: 80872601
	s_cselect_b32 s8, -1, 0                                    // 000000002FC8: 980880C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000002FCC: BF8704A9
	s_cmp_lg_u32 s8, 0                                         // 000000002FD0: BF078008
	s_sub_co_ci_u32 s6, s6, 0                                  // 000000002FD4: 82868006
	s_cmp_ge_u32 s6, s39                                       // 000000002FD8: BF092706
	s_cselect_b32 s8, -1, 0                                    // 000000002FDC: 980880C1
	s_cmp_ge_u32 s7, s38                                       // 000000002FE0: BF092607
	s_cselect_b32 s9, -1, 0                                    // 000000002FE4: 980980C1
	s_cmp_eq_u32 s6, s39                                       // 000000002FE8: BF062706
	s_add_nc_u64 s[6:7], s[2:3], 1                             // 000000002FEC: A9868102
	s_cselect_b32 s30, s9, s8                                  // 000000002FF0: 981E0809
	s_add_nc_u64 s[8:9], s[2:3], 2                             // 000000002FF4: A9888202
	s_cmp_lg_u32 s30, 0                                        // 000000002FF8: BF07801E
	s_cselect_b32 s6, s8, s6                                   // 000000002FFC: 98060608
	s_cselect_b32 s7, s9, s7                                   // 000000003000: 98070709
	s_cmp_lg_u32 s4, 0                                         // 000000003004: BF078004
	s_sub_co_ci_u32 s4, s37, s5                                // 000000003008: 82840525
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000300C: BF870009
	s_cmp_ge_u32 s4, s39                                       // 000000003010: BF092704
	s_cselect_b32 s5, -1, 0                                    // 000000003014: 980580C1
	s_cmp_ge_u32 s1, s38                                       // 000000003018: BF092601
	s_cselect_b32 s1, -1, 0                                    // 00000000301C: 980180C1
	s_cmp_eq_u32 s4, s39                                       // 000000003020: BF062704
	s_wait_alu 0xfffe                                          // 000000003024: BF88FFFE
	s_cselect_b32 s1, s1, s5                                   // 000000003028: 98010501
	s_wait_alu 0xfffe                                          // 00000000302C: BF88FFFE
	s_cmp_lg_u32 s1, 0                                         // 000000003030: BF078001
	s_cselect_b32 s3, s7, s3                                   // 000000003034: 98030307
	s_cselect_b32 s2, s6, s2                                   // 000000003038: 98020206
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000303C: 916A007E
	s_wait_alu 0xfffe                                          // 000000003040: BF88FFFE
	s_cbranch_vccnz 33                                         // 000000003044: BFA40021 <ullm_paged_decode_attn_f32_kernel+0x6cc>
	v_cvt_f32_u32_e32 v1, s38                                  // 000000003048: 7E020C26
	s_sub_co_i32 s1, 0, s38                                    // 00000000304C: 81812680
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003050: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000003054: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000003058: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003060: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000003064: 7E020F01
	v_readfirstlane_b32 s0, v1                                 // 000000003068: 7E000501
	s_wait_alu 0xfffe                                          // 00000000306C: BF88FFFE
	s_mul_i32 s1, s1, s0                                       // 000000003070: 96010001
	s_wait_alu 0xfffe                                          // 000000003074: BF88FFFE
	s_mul_hi_u32 s1, s0, s1                                    // 000000003078: 96810100
	s_wait_alu 0xfffe                                          // 00000000307C: BF88FFFE
	s_add_co_i32 s0, s0, s1                                    // 000000003080: 81000100
	s_wait_alu 0xfffe                                          // 000000003084: BF88FFFE
	s_mul_hi_u32 s0, s36, s0                                   // 000000003088: 96800024
	s_wait_alu 0xfffe                                          // 00000000308C: BF88FFFE
	s_mul_i32 s1, s0, s38                                      // 000000003090: 96012600
	s_add_co_i32 s2, s0, 1                                     // 000000003094: 81028100
	s_wait_alu 0xfffe                                          // 000000003098: BF88FFFE
	s_sub_co_i32 s1, s36, s1                                   // 00000000309C: 81810124
	s_wait_alu 0xfffe                                          // 0000000030A0: BF88FFFE
	s_sub_co_i32 s3, s1, s38                                   // 0000000030A4: 81832601
	s_cmp_ge_u32 s1, s38                                       // 0000000030A8: BF092601
	s_cselect_b32 s0, s2, s0                                   // 0000000030AC: 98000002
	s_wait_alu 0xfffe                                          // 0000000030B0: BF88FFFE
	s_cselect_b32 s1, s3, s1                                   // 0000000030B4: 98010103
	s_add_co_i32 s2, s0, 1                                     // 0000000030B8: 81028100
	s_wait_alu 0xfffe                                          // 0000000030BC: BF88FFFE
	s_cmp_ge_u32 s1, s38                                       // 0000000030C0: BF092601
	s_mov_b32 s3, 0                                            // 0000000030C4: BE830080
	s_cselect_b32 s2, s2, s0                                   // 0000000030C8: 98020002
	s_wait_alu 0xfffe                                          // 0000000030CC: BF88FFFE
	v_or_b32_e32 v5, s3, v7                                    // 0000000030D0: 380A0E03
	v_mov_b32_e32 v4, 0                                        // 0000000030D4: 7E080280
	s_mov_b32 s0, exec_lo                                      // 0000000030D8: BE80007E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000030DC: BF870001
	v_cmpx_ne_u64_e32 0, v[4:5]                                // 0000000030E0: 7DBA0880
	s_wait_alu 0xfffe                                          // 0000000030E4: BF88FFFE
	s_xor_b32 s1, exec_lo, s0                                  // 0000000030E8: 8D01007E
	s_cbranch_execz 164                                        // 0000000030EC: BFA500A4 <ullm_paged_decode_attn_f32_kernel+0x980>
	s_cvt_f32_u32 s0, s2                                       // 0000000030F0: BE806502
	s_cvt_f32_u32 s4, s3                                       // 0000000030F4: BE846503
	s_sub_nc_u64 s[6:7], 0, s[2:3]                             // 0000000030F8: AA060280
	s_mov_b32 s31, 0                                           // 0000000030FC: BE9F0080
	s_wait_alu 0xfffe                                          // 000000003100: BF88FFFE
	s_fmamk_f32 s0, s4, 0x4f800000, s0                         // 000000003104: A3000004 4F800000
	s_wait_alu 0xfffe                                          // 00000000310C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003110: BF87029A
	v_s_rcp_f32 s0, s0                                         // 000000003114: D6840000 00000000
	s_mul_f32 s0, s0, 0x5f7ffffc                               // 00000000311C: A200FF00 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000003124: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000003128: BF87059A
	s_mul_f32 s4, s0, 0x2f800000                               // 00000000312C: A204FF00 2F800000
	s_trunc_f32 s4, s4                                         // 000000003134: BE846204
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003138: BF8704BB
	s_fmamk_f32 s0, s4, 0xcf800000, s0                         // 00000000313C: A3000004 CF800000
	s_cvt_u32_f32 s5, s4                                       // 000000003144: BE856704
	s_wait_alu 0xfffe                                          // 000000003148: BF88FFFE
	s_cvt_u32_f32 s4, s0                                       // 00000000314C: BE846700
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003150: BF87049B
	s_mul_u64 s[8:9], s[6:7], s[4:5]                           // 000000003154: AA880406
	s_mul_hi_u32 s35, s4, s9                                   // 000000003158: 96A30904
	s_mul_i32 s34, s4, s9                                      // 00000000315C: 96220904
	s_mul_hi_u32 s30, s4, s8                                   // 000000003160: 969E0804
	s_mul_i32 s44, s5, s8                                      // 000000003164: 962C0805
	s_add_nc_u64 s[34:35], s[30:31], s[34:35]                  // 000000003168: A9A2221E
	s_mul_hi_u32 s0, s5, s8                                    // 00000000316C: 96800805
	s_mul_hi_u32 s45, s5, s9                                   // 000000003170: 96AD0905
	s_mul_i32 s8, s5, s9                                       // 000000003174: 96080905
	s_add_co_u32 s9, s34, s44                                  // 000000003178: 80092C22
	s_wait_alu 0xfffe                                          // 00000000317C: BF88FFFE
	s_add_co_ci_u32 s30, s35, s0                               // 000000003180: 821E0023
	s_add_co_ci_u32 s9, s45, 0                                 // 000000003184: 8209802D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003188: BF870499
	s_add_nc_u64 s[8:9], s[30:31], s[8:9]                      // 00000000318C: A988081E
	s_add_co_u32 s4, s4, s8                                    // 000000003190: 80040804
	s_cselect_b32 s0, -1, 0                                    // 000000003194: 980080C1
	s_wait_alu 0xfffe                                          // 000000003198: BF88FFFE
	s_cmp_lg_u32 s0, 0                                         // 00000000319C: BF078000
	s_add_co_ci_u32 s5, s5, s9                                 // 0000000031A0: 82050905
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000031A4: BF870499
	s_mul_u64 s[6:7], s[6:7], s[4:5]                           // 0000000031A8: AA860406
	s_mul_hi_u32 s9, s4, s7                                    // 0000000031AC: 96890704
	s_mul_i32 s8, s4, s7                                       // 0000000031B0: 96080704
	s_mul_hi_u32 s30, s4, s6                                   // 0000000031B4: 969E0604
	s_mul_i32 s34, s5, s6                                      // 0000000031B8: 96220605
	s_add_nc_u64 s[8:9], s[30:31], s[8:9]                      // 0000000031BC: A988081E
	s_mul_hi_u32 s0, s5, s6                                    // 0000000031C0: 96800605
	s_mul_hi_u32 s35, s5, s7                                   // 0000000031C4: 96A30705
	s_mul_i32 s6, s5, s7                                       // 0000000031C8: 96060705
	s_add_co_u32 s7, s8, s34                                   // 0000000031CC: 80072208
	s_wait_alu 0xfffe                                          // 0000000031D0: BF88FFFE
	s_add_co_ci_u32 s30, s9, s0                                // 0000000031D4: 821E0009
	s_add_co_ci_u32 s7, s35, 0                                 // 0000000031D8: 82078023
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000031DC: BF870499
	s_add_nc_u64 s[6:7], s[30:31], s[6:7]                      // 0000000031E0: A986061E
	s_add_co_u32 s0, s4, s6                                    // 0000000031E4: 80000604
	s_cselect_b32 s4, -1, 0                                    // 0000000031E8: 980480C1
	s_wait_alu 0xfffe                                          // 0000000031EC: BF88FFFE
	v_mul_hi_u32 v1, v6, s0                                    // 0000000031F0: D72D0001 00000106
	s_cmp_lg_u32 s4, 0                                         // 0000000031F8: BF078004
	v_mad_co_u64_u32 v[8:9], null, v7, s0, 0                   // 0000000031FC: D6FE7C08 02000107
	s_add_co_ci_u32 s4, s5, s7                                 // 000000003204: 82040705
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003208: BF870129
	v_mad_co_u64_u32 v[4:5], null, v6, s4, 0                   // 00000000320C: D6FE7C04 02000906
	v_mad_co_u64_u32 v[10:11], null, v7, s4, 0                 // 000000003214: D6FE7C0A 02000907
	v_add_co_u32 v1, vcc_lo, v1, v4                            // 00000000321C: D7006A01 00020901
	s_wait_alu 0xfffd                                          // 000000003224: BF88FFFD
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003228: BF870113
	v_add_co_ci_u32_e64 v4, null, 0, v5, vcc_lo                // 00000000322C: D5207C04 01AA0A80
	v_add_co_u32 v1, vcc_lo, v1, v8                            // 000000003234: D7006A01 00021101
	s_wait_alu 0xfffd                                          // 00000000323C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000003240: BF870132
	v_add_co_ci_u32_e32 v1, vcc_lo, v4, v9, vcc_lo             // 000000003244: 40021304
	s_wait_alu 0xfffd                                          // 000000003248: BF88FFFD
	v_add_co_ci_u32_e32 v4, vcc_lo, 0, v11, vcc_lo             // 00000000324C: 40081680
	v_add_co_u32 v1, vcc_lo, v1, v10                           // 000000003250: D7006A01 00021501
	s_wait_alu 0xfffd                                          // 000000003258: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000325C: BF870112
	v_add_co_ci_u32_e64 v8, null, 0, v4, vcc_lo                // 000000003260: D5207C08 01AA0880
	v_mul_lo_u32 v9, s3, v1                                    // 000000003268: D72C0009 00020203
	v_mad_co_u64_u32 v[4:5], null, s2, v1, 0                   // 000000003270: D6FE7C04 02020202
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003278: BF870113
	v_mul_lo_u32 v10, s2, v8                                   // 00000000327C: D72C000A 00021002
	v_sub_co_u32 v4, vcc_lo, v6, v4                            // 000000003284: D7016A04 00020906
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 00000000328C: BF8701C2
	v_add3_u32 v5, v5, v10, v9                                 // 000000003290: D6550005 04261505
	v_add_co_u32 v10, s0, v1, 2                                // 000000003298: D700000A 00010501
	s_wait_alu 0xf1ff                                          // 0000000032A0: BF88F1FF
	v_add_co_ci_u32_e64 v11, null, 0, v8, s0                   // 0000000032A4: D5207C0B 00021080
	v_sub_nc_u32_e32 v9, v7, v5                                // 0000000032AC: 4C120B07
	v_sub_co_u32 v12, s0, v4, s2                               // 0000000032B0: D701000C 00000504
	s_wait_alu 0xfffd                                          // 0000000032B8: BF88FFFD
	v_sub_co_ci_u32_e64 v5, null, v7, v5, vcc_lo               // 0000000032BC: D5217C05 01AA0B07
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000032C4: BF870193
	v_subrev_co_ci_u32_e64 v9, null, s3, v9, vcc_lo            // 0000000032C8: D5227C09 01AA1203
	v_cmp_le_u32_e32 vcc_lo, s2, v12                           // 0000000032D0: 7C961802
	s_wait_alu 0xf1ff                                          // 0000000032D4: BF88F1FF
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000032D8: BF8701C2
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, s0                 // 0000000032DC: D5227C09 00021280
	s_wait_alu 0xfffd                                          // 0000000032E4: BF88FFFD
	v_cndmask_b32_e64 v12, 0, -1, vcc_lo                       // 0000000032E8: D501000C 01A98280
	v_cmp_eq_u32_e64 s0, s3, v5                                // 0000000032F0: D44A0000 00020A03
	v_cmp_le_u32_e32 vcc_lo, s3, v9                            // 0000000032F8: 7C961203
	s_wait_alu 0xfffd                                          // 0000000032FC: BF88FFFD
	v_cndmask_b32_e64 v13, 0, -1, vcc_lo                       // 000000003300: D501000D 01A98280
	v_cmp_le_u32_e32 vcc_lo, s2, v4                            // 000000003308: 7C960802
	s_wait_alu 0xfffd                                          // 00000000330C: BF88FFFD
	v_cndmask_b32_e64 v4, 0, -1, vcc_lo                        // 000000003310: D5010004 01A98280
	v_cmp_le_u32_e32 vcc_lo, s3, v5                            // 000000003318: 7C960A03
	s_wait_alu 0xfffd                                          // 00000000331C: BF88FFFD
	v_cndmask_b32_e64 v14, 0, -1, vcc_lo                       // 000000003320: D501000E 01A98280
	v_cmp_eq_u32_e32 vcc_lo, s3, v9                            // 000000003328: 7C941203
	s_wait_alu 0xf1ff                                          // 00000000332C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003330: BF870002
	v_cndmask_b32_e64 v4, v14, v4, s0                          // 000000003334: D5010004 0002090E
	s_wait_alu 0xfffd                                          // 00000000333C: BF88FFFD
	v_cndmask_b32_e32 v9, v13, v12, vcc_lo                     // 000000003340: 0212190D
	v_add_co_u32 v12, vcc_lo, v1, 1                            // 000000003344: D7006A0C 00010301
	s_wait_alu 0xfffd                                          // 00000000334C: BF88FFFD
	v_add_co_ci_u32_e64 v13, null, 0, v8, vcc_lo               // 000000003350: D5207C0D 01AA1080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003358: BF870123
	v_cmp_ne_u32_e32 vcc_lo, 0, v9                             // 00000000335C: 7C9A1280
	s_wait_alu 0xfffd                                          // 000000003360: BF88FFFD
	v_dual_cndmask_b32 v5, v13, v11 :: v_dual_cndmask_b32 v10, v12, v10// 000000003364: CA52170D 050A150C
	v_cmp_ne_u32_e32 vcc_lo, 0, v4                             // 00000000336C: 7C9A0880
	s_wait_alu 0xfffd                                          // 000000003370: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003374: BF870002
	v_dual_cndmask_b32 v9, v8, v5 :: v_dual_cndmask_b32 v8, v1, v10// 000000003378: CA520B08 09081501
	s_wait_alu 0xfffe                                          // 000000003380: BF88FFFE
	s_and_not1_saveexec_b32 s0, s1                             // 000000003384: BE803001
	s_cbranch_execz 35                                         // 000000003388: BFA50023 <ullm_paged_decode_attn_f32_kernel+0xa18>
	v_cvt_f32_u32_e32 v1, s2                                   // 00000000338C: 7E020C02
	s_sub_co_i32 s1, 0, s2                                     // 000000003390: 81810280
	v_mov_b32_e32 v9, 0                                        // 000000003394: 7E120280
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003398: BF870292
	v_rcp_iflag_f32_e32 v1, v1                                 // 00000000339C: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000033A0: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000033A8: BF8700A1
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000033AC: 7E020F01
	s_wait_alu 0xfffe                                          // 0000000033B0: BF88FFFE
	v_mul_lo_u32 v4, s1, v1                                    // 0000000033B4: D72C0004 00020201
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033BC: BF870091
	v_mul_hi_u32 v4, v1, v4                                    // 0000000033C0: D72D0004 00020901
	v_add_nc_u32_e32 v1, v1, v4                                // 0000000033C8: 4A020901
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033CC: BF870091
	v_mul_hi_u32 v1, v6, v1                                    // 0000000033D0: D72D0001 00020306
	v_mul_lo_u32 v4, v1, s2                                    // 0000000033D8: D72C0004 00000501
	v_add_nc_u32_e32 v5, 1, v1                                 // 0000000033E0: 4A0A0281
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033E4: BF870092
	v_sub_nc_u32_e32 v4, v6, v4                                // 0000000033E8: 4C080906
	v_subrev_nc_u32_e32 v8, s2, v4                             // 0000000033EC: 4E100802
	v_cmp_le_u32_e32 vcc_lo, s2, v4                            // 0000000033F0: 7C960802
	s_wait_alu 0xfffd                                          // 0000000033F4: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033F8: BF870092
	v_dual_cndmask_b32 v4, v4, v8 :: v_dual_cndmask_b32 v1, v1, v5// 0000000033FC: CA521104 04000B01
	v_cmp_le_u32_e32 vcc_lo, s2, v4                            // 000000003404: 7C960802
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003408: BF8700A2
	v_add_nc_u32_e32 v5, 1, v1                                 // 00000000340C: 4A0A0281
	s_wait_alu 0xfffd                                          // 000000003410: BF88FFFD
	v_cndmask_b32_e32 v8, v1, v5, vcc_lo                       // 000000003414: 02100B01
	s_wait_alu 0xfffe                                          // 000000003418: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000341C: 8C7E007E
	v_mul_lo_u32 v1, v7, s40                                   // 000000003420: D72C0001 00005107
	v_mul_lo_u32 v12, v6, s41                                  // 000000003428: D72C000C 00005306
	v_mad_co_u64_u32 v[10:11], null, v6, s40, 0                // 000000003430: D6FE7C0A 02005106
	v_lshlrev_b64_e32 v[4:5], 2, v[2:3]                        // 000000003438: 3E080482
	s_cmp_lg_u64 s[22:23], 0                                   // 00000000343C: BF118016
	s_mov_b64 s[0:1], 0                                        // 000000003440: BE800180
	s_cselect_b32 s34, -1, 0                                   // 000000003444: 982280C1
	s_cmp_eq_u64 s[22:23], 0                                   // 000000003448: BF108016
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000344C: BF870002
	v_add3_u32 v11, v11, v12, v1                               // 000000003450: D655000B 0406190B
	s_cbranch_scc1 289                                         // 000000003458: BFA20121 <ullm_paged_decode_attn_f32_kernel+0xee0>
	s_cvt_f32_u32 s2, s24                                      // 00000000345C: BE826518
	s_cvt_f32_u32 s3, s25                                      // 000000003460: BE836519
	v_cvt_f32_u32_e32 v1, s24                                  // 000000003464: 7E020C18
	v_lshlrev_b64_e32 v[15:16], 2, v[10:11]                    // 000000003468: 3E1E1482
	v_add_co_u32 v12, vcc_lo, s28, v4                          // 00000000346C: D7006A0C 0002081C
	s_wait_alu 0xfffe                                          // 000000003474: BF88FFFE
	s_fmamk_f32 s2, s3, 0x4f800000, s2                         // 000000003478: A3020203 4F800000
	v_rcp_iflag_f32_e32 v14, v1                                // 000000003480: 7E1C5701
	s_wait_alu 0xfffd                                          // 000000003484: BF88FFFD
	v_add_co_ci_u32_e64 v13, null, s29, v5, vcc_lo             // 000000003488: D5207C0D 01AA0A1D
	s_wait_alu 0xfffe                                          // 000000003490: BF88FFFE
	v_s_rcp_f32 s2, s2                                         // 000000003494: D6840002 00000002
	v_add_co_u32 v15, vcc_lo, s12, v15                         // 00000000349C: D7006A0F 00021E0C
	v_dual_mov_b32 v18, 0xff7fffff :: v_dual_mov_b32 v1, 0     // 0000000034A4: CA1000FF 12000080 FF7FFFFF
	s_wait_alu 0xfffd                                          // 0000000034B0: BF88FFFD
	v_add_co_ci_u32_e64 v16, null, s13, v16, vcc_lo            // 0000000034B4: D5207C10 01AA200D
	s_delay_alu instid0(TRANS32_DEP_1)                         // 0000000034BC: BF870005
	s_mul_f32 s2, s2, 0x5f7ffffc                               // 0000000034C0: A202FF02 5F7FFFFC
	v_mul_f32_e32 v14, 0x4f7ffffe, v14                         // 0000000034C8: 101C1CFF 4F7FFFFE
	s_cmp_lg_u64 s[40:41], 0                                   // 0000000034D0: BF118028
	s_mov_b32 s3, 0                                            // 0000000034D4: BE830080
	s_wait_alu 0xfffe                                          // 0000000034D8: BF88FFFE
	s_mul_f32 s4, s2, 0x2f800000                               // 0000000034DC: A204FF02 2F800000
	s_cselect_b32 s35, -1, 0                                   // 0000000034E4: 982380C1
	v_cvt_u32_f32_e32 v17, v14                                 // 0000000034E8: 7E220F0E
	s_sub_nc_u64 s[6:7], 0, s[24:25]                           // 0000000034EC: AA061880
	s_wait_alu 0xfffe                                          // 0000000034F0: BF88FFFE
	s_trunc_f32 s5, s4                                         // 0000000034F4: BE856204
	s_sub_co_i32 s44, 0, s24                                   // 0000000034F8: 81AC1880
	s_wait_alu 0xfffe                                          // 0000000034FC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003500: BF8704B9
	s_fmamk_f32 s2, s5, 0xcf800000, s2                         // 000000003504: A3020205 CF800000
	s_cvt_u32_f32 s5, s5                                       // 00000000350C: BE856705
	s_wait_alu 0xfffe                                          // 000000003510: BF88FFFE
	s_cvt_u32_f32 s4, s2                                       // 000000003514: BE846702
	s_or_b64 s[8:9], s[0:1], s[24:25]                          // 000000003518: 8C881800
	s_mov_b32 s8, s3                                           // 00000000351C: BE880003
	s_mov_b32 s2, -1                                           // 000000003520: BE8200C1
	s_wait_alu 0xfffe                                          // 000000003524: BF88FFFE
	s_cmp_lg_u64 s[8:9], 0                                     // 000000003528: BF118008
	s_cbranch_scc0 104                                         // 00000000352C: BFA10068 <ullm_paged_decode_attn_f32_kernel+0xcd0>
	s_mul_u64 s[8:9], s[6:7], s[4:5]                           // 000000003530: AA880406
	s_wait_alu 0xfffe                                          // 000000003534: BF88FFFE
	s_mul_hi_u32 s31, s4, s9                                   // 000000003538: 969F0904
	s_mul_i32 s30, s4, s9                                      // 00000000353C: 961E0904
	s_mul_hi_u32 s2, s4, s8                                    // 000000003540: 96820804
	s_mul_hi_u32 s45, s5, s9                                   // 000000003544: 96AD0905
	s_wait_alu 0xfffe                                          // 000000003548: BF88FFFE
	s_add_nc_u64 s[30:31], s[2:3], s[30:31]                    // 00000000354C: A99E1E02
	s_mul_hi_u32 s2, s5, s8                                    // 000000003550: 96820805
	s_mul_i32 s8, s5, s8                                       // 000000003554: 96080805
	s_wait_alu 0xfffe                                          // 000000003558: BF88FFFE
	s_add_co_u32 s8, s30, s8                                   // 00000000355C: 8008081E
	s_add_co_ci_u32 s2, s31, s2                                // 000000003560: 8202021F
	s_add_co_ci_u32 s31, s45, 0                                // 000000003564: 821F802D
	s_mul_i32 s30, s5, s9                                      // 000000003568: 961E0905
	s_wait_alu 0xfffe                                          // 00000000356C: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[30:31]                      // 000000003570: A9881E02
	s_wait_alu 0xfffe                                          // 000000003574: BF88FFFE
	s_add_co_u32 s8, s4, s8                                    // 000000003578: 80080804
	s_cselect_b32 s2, -1, 0                                    // 00000000357C: 980280C1
	s_wait_alu 0xfffe                                          // 000000003580: BF88FFFE
	s_cmp_lg_u32 s2, 0                                         // 000000003584: BF078002
	s_add_co_ci_u32 s9, s5, s9                                 // 000000003588: 82090905
	s_wait_alu 0xfffe                                          // 00000000358C: BF88FFFE
	s_mul_u64 s[30:31], s[6:7], s[8:9]                         // 000000003590: AA9E0806
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003594: BF870009
	s_mul_hi_u32 s47, s8, s31                                  // 000000003598: 96AF1F08
	s_mul_i32 s46, s8, s31                                     // 00000000359C: 962E1F08
	s_mul_hi_u32 s2, s8, s30                                   // 0000000035A0: 96821E08
	s_mul_hi_u32 s45, s9, s30                                  // 0000000035A4: 96AD1E09
	s_mul_i32 s30, s9, s30                                     // 0000000035A8: 961E1E09
	s_wait_alu 0xfffe                                          // 0000000035AC: BF88FFFE
	s_add_nc_u64 s[46:47], s[2:3], s[46:47]                    // 0000000035B0: A9AE2E02
	s_mul_hi_u32 s48, s9, s31                                  // 0000000035B4: 96B01F09
	s_wait_alu 0xfffe                                          // 0000000035B8: BF88FFFE
	s_add_co_u32 s2, s46, s30                                  // 0000000035BC: 80021E2E
	s_add_co_ci_u32 s2, s47, s45                               // 0000000035C0: 82022D2F
	s_mul_i32 s30, s9, s31                                     // 0000000035C4: 961E1F09
	s_add_co_ci_u32 s31, s48, 0                                // 0000000035C8: 821F8030
	s_wait_alu 0xfffe                                          // 0000000035CC: BF88FFFE
	s_add_nc_u64 s[30:31], s[2:3], s[30:31]                    // 0000000035D0: A99E1E02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000035D4: BF870009
	s_add_co_u32 s8, s8, s30                                   // 0000000035D8: 80081E08
	s_cselect_b32 s30, -1, 0                                   // 0000000035DC: 981E80C1
	s_wait_alu 0xfffe                                          // 0000000035E0: BF88FFFE
	s_mul_hi_u32 s2, s0, s8                                    // 0000000035E4: 96820800
	s_cmp_lg_u32 s30, 0                                        // 0000000035E8: BF07801E
	s_mul_hi_u32 s45, s1, s8                                   // 0000000035EC: 96AD0801
	s_add_co_ci_u32 s30, s9, s31                               // 0000000035F0: 821E1F09
	s_mul_i32 s31, s1, s8                                      // 0000000035F4: 961F0801
	s_mul_hi_u32 s9, s0, s30                                   // 0000000035F8: 96891E00
	s_mul_i32 s8, s0, s30                                      // 0000000035FC: 96081E00
	s_mul_hi_u32 s46, s1, s30                                  // 000000003600: 96AE1E01
	s_wait_alu 0xfffe                                          // 000000003604: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[8:9]                        // 000000003608: A9880802
	s_mul_i32 s30, s1, s30                                     // 00000000360C: 961E1E01
	s_wait_alu 0xfffe                                          // 000000003610: BF88FFFE
	s_add_co_u32 s2, s8, s31                                   // 000000003614: 80021F08
	s_add_co_ci_u32 s2, s9, s45                                // 000000003618: 82022D09
	s_add_co_ci_u32 s31, s46, 0                                // 00000000361C: 821F802E
	s_wait_alu 0xfffe                                          // 000000003620: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[30:31]                      // 000000003624: A9881E02
	s_wait_alu 0xfffe                                          // 000000003628: BF88FFFE
	s_mul_u64 s[30:31], s[24:25], s[8:9]                       // 00000000362C: AA9E0818
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003630: BF870009
	s_sub_co_u32 s2, s0, s30                                   // 000000003634: 80821E00
	s_cselect_b32 s30, -1, 0                                   // 000000003638: 981E80C1
	s_sub_co_i32 s45, s1, s31                                  // 00000000363C: 81AD1F01
	s_cmp_lg_u32 s30, 0                                        // 000000003640: BF07801E
	s_sub_co_ci_u32 s45, s45, s25                              // 000000003644: 82AD192D
	s_wait_alu 0xfffe                                          // 000000003648: BF88FFFE
	s_sub_co_u32 s46, s2, s24                                  // 00000000364C: 80AE1802
	s_cselect_b32 s47, -1, 0                                   // 000000003650: 982F80C1
	s_wait_alu 0xfffe                                          // 000000003654: BF88FFFE
	s_cmp_lg_u32 s47, 0                                        // 000000003658: BF07802F
	s_sub_co_ci_u32 s45, s45, 0                                // 00000000365C: 82AD802D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003660: BF870009
	s_cmp_ge_u32 s45, s25                                      // 000000003664: BF09192D
	s_cselect_b32 s48, -1, 0                                   // 000000003668: 983080C1
	s_cmp_ge_u32 s46, s24                                      // 00000000366C: BF09182E
	s_add_nc_u64 s[46:47], s[8:9], 1                           // 000000003670: A9AE8108
	s_cselect_b32 s49, -1, 0                                   // 000000003674: 983180C1
	s_cmp_eq_u32 s45, s25                                      // 000000003678: BF06192D
	s_cselect_b32 s45, s49, s48                                // 00000000367C: 982D3031
	s_add_nc_u64 s[48:49], s[8:9], 2                           // 000000003680: A9B08208
	s_cmp_lg_u32 s45, 0                                        // 000000003684: BF07802D
	s_wait_alu 0xfffe                                          // 000000003688: BF88FFFE
	s_cselect_b32 s45, s48, s46                                // 00000000368C: 982D2E30
	s_cselect_b32 s46, s49, s47                                // 000000003690: 982E2F31
	s_cmp_lg_u32 s30, 0                                        // 000000003694: BF07801E
	s_sub_co_ci_u32 s30, s1, s31                               // 000000003698: 829E1F01
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000369C: BF870009
	s_cmp_ge_u32 s30, s25                                      // 0000000036A0: BF09191E
	s_cselect_b32 s31, -1, 0                                   // 0000000036A4: 981F80C1
	s_cmp_ge_u32 s2, s24                                       // 0000000036A8: BF091802
	s_cselect_b32 s2, -1, 0                                    // 0000000036AC: 980280C1
	s_cmp_eq_u32 s30, s25                                      // 0000000036B0: BF06191E
	s_wait_alu 0xfffe                                          // 0000000036B4: BF88FFFE
	s_cselect_b32 s2, s2, s31                                  // 0000000036B8: 98021F02
	s_wait_alu 0xfffe                                          // 0000000036BC: BF88FFFE
	s_cmp_lg_u32 s2, 0                                         // 0000000036C0: BF078002
	s_mov_b32 s2, 0                                            // 0000000036C4: BE820080
	s_cselect_b32 s9, s46, s9                                  // 0000000036C8: 9809092E
	s_cselect_b32 s8, s45, s8                                  // 0000000036CC: 9808082D
	s_wait_alu 0xfffe                                          // 0000000036D0: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000036D4: 916A027E
	s_wait_alu 0xfffe                                          // 0000000036D8: BF88FFFE
	s_cbranch_vccnz 25                                         // 0000000036DC: BFA40019 <ullm_paged_decode_attn_f32_kernel+0xd44>
	v_readfirstlane_b32 s2, v17                                // 0000000036E0: 7E040511
	s_mul_i32 s8, s44, s2                                      // 0000000036E4: 9608022C
	s_wait_alu 0xfffe                                          // 0000000036E8: BF88FFFE
	s_mul_hi_u32 s8, s2, s8                                    // 0000000036EC: 96880802
	s_wait_alu 0xfffe                                          // 0000000036F0: BF88FFFE
	s_add_co_i32 s2, s2, s8                                    // 0000000036F4: 81020802
	s_wait_alu 0xfffe                                          // 0000000036F8: BF88FFFE
	s_mul_hi_u32 s2, s0, s2                                    // 0000000036FC: 96820200
	s_wait_alu 0xfffe                                          // 000000003700: BF88FFFE
	s_mul_i32 s8, s2, s24                                      // 000000003704: 96081802
	s_add_co_i32 s9, s2, 1                                     // 000000003708: 81098102
	s_wait_alu 0xfffe                                          // 00000000370C: BF88FFFE
	s_sub_co_i32 s8, s0, s8                                    // 000000003710: 81880800
	s_wait_alu 0xfffe                                          // 000000003714: BF88FFFE
	s_sub_co_i32 s30, s8, s24                                  // 000000003718: 819E1808
	s_cmp_ge_u32 s8, s24                                       // 00000000371C: BF091808
	s_cselect_b32 s2, s9, s2                                   // 000000003720: 98020209
	s_cselect_b32 s8, s30, s8                                  // 000000003724: 9808081E
	s_wait_alu 0xfffe                                          // 000000003728: BF88FFFE
	s_add_co_i32 s9, s2, 1                                     // 00000000372C: 81098102
	s_cmp_ge_u32 s8, s24                                       // 000000003730: BF091808
	s_wait_alu 0xfffe                                          // 000000003734: BF88FFFE
	s_cselect_b32 s2, s9, s2                                   // 000000003738: 98020209
	s_wait_alu 0xfffe                                          // 00000000373C: BF88FFFE
	s_mov_b64 s[8:9], s[2:3]                                   // 000000003740: BE880102
	s_wait_alu 0xfffe                                          // 000000003744: BF88FFFE
	s_lshl_b64 s[30:31], s[8:9], 2                             // 000000003748: 849E8208
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000374C: BF870009
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]                  // 000000003750: A99E1E14
	s_load_b32 s2, s[30:31], 0x0                               // 000000003754: F400008F F8000000
	s_wait_kmcnt 0x0                                           // 00000000375C: BFC70000
	v_cmp_gt_u64_e64 s30, s[26:27], s[2:3]                     // 000000003760: D45C001E 0000041A
	v_cmp_le_u64_e64 s45, s[26:27], s[2:3]                     // 000000003768: D45B002D 0000041A
	s_and_b32 vcc_lo, exec_lo, s30                             // 000000003770: 8B6A1E7E
	s_mov_b32 s30, -1                                          // 000000003774: BE9E00C1
	s_wait_alu 0xfffe                                          // 000000003778: BF88FFFE
	s_cbranch_vccz 68                                          // 00000000377C: BFA30044 <ullm_paged_decode_attn_f32_kernel+0xe90>
	s_and_not1_b32 vcc_lo, exec_lo, s35                        // 000000003780: 916A237E
	s_wait_alu 0xfffe                                          // 000000003784: BF88FFFE
	s_cbranch_vccnz 58                                         // 000000003788: BFA4003A <ullm_paged_decode_attn_f32_kernel+0xe74>
	s_sub_nc_u64 s[8:9], s[2:3], s[8:9]                        // 00000000378C: AA080802
	s_mov_b64 s[30:31], s[40:41]                               // 000000003790: BE9E0128
	s_wait_alu 0xfffe                                          // 000000003794: BF88FFFE
	s_mul_u64 s[8:9], s[8:9], s[24:25]                         // 000000003798: AA881808
	s_wait_alu 0xfffe                                          // 00000000379C: BF88FFFE
	s_add_nc_u64 s[8:9], s[8:9], s[0:1]                        // 0000000037A0: A9880008
	s_wait_alu 0xfffe                                          // 0000000037A4: BF88FFFE
	v_mad_co_u64_u32 v[19:20], null, s8, s38, v[8:9]           // 0000000037A8: D6FE7C13 04204C08
	s_mul_i32 s2, s9, s38                                      // 0000000037B0: 96022609
	s_mul_i32 s8, s8, s39                                      // 0000000037B4: 96082708
	s_wait_alu 0xfffe                                          // 0000000037B8: BF88FFFE
	v_add3_u32 v14, s8, s2, v20                                // 0000000037BC: D655000E 04500408
	v_mul_lo_u32 v21, v19, s41                                 // 0000000037C4: D72C0015 00005313
	v_mad_co_u64_u32 v[19:20], null, v19, s40, 0               // 0000000037CC: D6FE7C13 02005113
	s_mov_b64 s[8:9], 0                                        // 0000000037D4: BE880180
	v_mul_lo_u32 v14, v14, s40                                 // 0000000037D8: D72C000E 0000510E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000037E0: BF870121
	v_add3_u32 v20, v20, v21, v14                              // 0000000037E4: D6550014 043A2B14
	v_mov_b32_e32 v14, 0                                       // 0000000037EC: 7E1C0280
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]                    // 0000000037F0: 3E262682
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000037F4: BF870121
	v_add_co_u32 v19, vcc_lo, s16, v19                         // 0000000037F8: D7006A13 00022610
	s_wait_alu 0xfffd                                          // 000000003800: BF88FFFD
	v_add_co_ci_u32_e64 v20, null, s17, v20, vcc_lo            // 000000003804: D5207C14 01AA2811
	s_wait_alu 0xfffe                                          // 00000000380C: BF88FFFE
	s_lshl_b64 s[46:47], s[8:9], 2                             // 000000003810: 84AE8208
	s_add_nc_u64 s[30:31], s[30:31], -1                        // 000000003814: A99EC11E
	s_wait_alu 0xfffe                                          // 000000003818: BF88FFFE
	v_add_co_u32 v21, vcc_lo, v15, s46                         // 00000000381C: D7006A15 00005D0F
	s_wait_alu 0xfffd                                          // 000000003824: BF88FFFD
	v_add_co_ci_u32_e64 v22, null, s47, v16, vcc_lo            // 000000003828: D5207C16 01AA202F
	v_add_co_u32 v23, vcc_lo, v19, s46                         // 000000003830: D7006A17 00005D13
	s_wait_alu 0xfffd                                          // 000000003838: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s47, v20, vcc_lo            // 00000000383C: D5207C18 01AA282F
	global_load_b32 v21, v[21:22], off                         // 000000003844: EE05007C 00000015 00000015
	global_load_b32 v22, v[23:24], off                         // 000000003850: EE05007C 00000016 00000017
	s_cmp_eq_u64 s[30:31], 0                                   // 00000000385C: BF10801E
	s_add_nc_u64 s[8:9], s[8:9], 1                             // 000000003860: A9888108
	s_wait_loadcnt 0x0                                         // 000000003864: BFC00000
	v_fmac_f32_e32 v14, v21, v22                               // 000000003868: 561C2D15
	s_cbranch_scc0 65511                                       // 00000000386C: BFA1FFE7 <ullm_paged_decode_attn_f32_kernel+0xe0c>
	s_branch 1                                                 // 000000003870: BFA00001 <ullm_paged_decode_attn_f32_kernel+0xe78>
	v_mov_b32_e32 v14, 0                                       // 000000003874: 7E1C0280
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003878: BF8700A1
	v_mul_f32_e32 v14, s33, v14                                // 00000000387C: 101C1C21
	s_mov_b32 s30, 0                                           // 000000003880: BE9E0080
	v_cmp_gt_f32_e32 vcc_lo, v14, v18                          // 000000003884: 7C28250E
	s_wait_alu 0xfffd                                          // 000000003888: BF88FFFD
	v_cndmask_b32_e32 v14, v18, v14, vcc_lo                    // 00000000388C: 021C1D12
	s_and_b32 vcc_lo, exec_lo, s30                             // 000000003890: 8B6A1E7E
	s_wait_alu 0xfffe                                          // 000000003894: BF88FFFE
	s_cbranch_vccz 4                                           // 000000003898: BFA30004 <ullm_paged_decode_attn_f32_kernel+0xeac>
	v_mov_b32_e32 v14, v18                                     // 00000000389C: 7E1C0312
	global_store_b32 v[12:13], v1, off                         // 0000000038A0: EE06807C 00800000 0000000C
	s_add_nc_u64 s[0:1], s[0:1], 1                             // 0000000038AC: A9808100
	s_wait_alu 0xfffe                                          // 0000000038B0: BF88FFFE
	s_cmp_eq_u64 s[0:1], s[22:23]                              // 0000000038B4: BF101600
	s_cselect_b32 s2, -1, 0                                    // 0000000038B8: 980280C1
	s_wait_alu 0xfffe                                          // 0000000038BC: BF88FFFE
	s_or_b32 s2, s45, s2                                       // 0000000038C0: 8C02022D
	s_wait_alu 0xfffe                                          // 0000000038C4: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000038C8: 916A027E
	s_wait_alu 0xfffe                                          // 0000000038CC: BF88FFFE
	s_cbranch_vccz 6                                           // 0000000038D0: BFA30006 <ullm_paged_decode_attn_f32_kernel+0xeec>
	v_mov_b32_e32 v18, v14                                     // 0000000038D4: 7E24030E
	s_branch 65295                                             // 0000000038D8: BFA0FF0F <ullm_paged_decode_attn_f32_kernel+0xb18>
	s_branch 64986                                             // 0000000038DC: BFA0FDDA <ullm_paged_decode_attn_f32_kernel+0x648>
	v_mov_b32_e32 v14, 0xff7fffff                              // 0000000038E0: 7E1C02FF FF7FFFFF
	s_mov_b32 s45, 0                                           // 0000000038E8: BEAD0080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000038EC: BF870009
	s_and_b32 vcc_lo, exec_lo, s45                             // 0000000038F0: 8B6A2D7E
	s_wait_alu 0xfffe                                          // 0000000038F4: BF88FFFE
	s_cbranch_vccnz 471                                        // 0000000038F8: BFA401D7 <ullm_paged_decode_attn_f32_kernel+0x1658>
	s_and_not1_b32 vcc_lo, exec_lo, s34                        // 0000000038FC: 916A227E
	s_wait_alu 0xfffe                                          // 000000003900: BF88FFFE
	s_cbranch_vccnz 356                                        // 000000003904: BFA40164 <ullm_paged_decode_attn_f32_kernel+0x1498>
	v_mul_lo_u32 v1, v7, s42                                   // 000000003908: D72C0001 00005507
	v_mul_lo_u32 v12, v6, s43                                  // 000000003910: D72C000C 00005706
	v_mad_co_u64_u32 v[6:7], null, v6, s42, 0                  // 000000003918: D6FE7C06 02005506
	s_cvt_f32_u32 s0, s24                                      // 000000003920: BE806518
	s_cvt_f32_u32 s1, s25                                      // 000000003924: BE816519
	v_cvt_f32_u32_e32 v13, s24                                 // 000000003928: 7E1A0C18
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0             // 00000000392C: CA100080 10100080
	s_wait_alu 0xfffe                                          // 000000003934: BF88FFFE
	s_fmamk_f32 s0, s1, 0x4f800000, s0                         // 000000003938: A3000001 4F800000
	v_add3_u32 v7, v7, v12, v1                                 // 000000003940: D6550007 04061907
	v_sub_co_u32 v1, vcc_lo, v2, v6                            // 000000003948: D7016A01 00020D02
	s_wait_alu 0xfffe                                          // 000000003950: BF88FFFE
	v_s_rcp_f32 s0, s0                                         // 000000003954: D6840000 00000000
	v_lshlrev_b64_e32 v[11:12], 2, v[10:11]                    // 00000000395C: 3E161482
	s_wait_alu 0xfffd                                          // 000000003960: BF88FFFD
	v_sub_co_ci_u32_e64 v2, null, v3, v7, vcc_lo               // 000000003964: D5217C02 01AA0F03
	s_cmp_lg_u64 s[40:41], 0                                   // 00000000396C: BF118028
	s_sub_nc_u64 s[6:7], 0, s[24:25]                           // 000000003970: AA061880
	s_cselect_b32 s34, -1, 0                                   // 000000003974: 982280C1
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]                        // 000000003978: 3E0C0282
	v_add_co_u32 v1, vcc_lo, s28, v4                           // 00000000397C: D7006A01 0002081C
	s_wait_alu 0xfffd                                          // 000000003984: BF88FFFD
	v_add_co_ci_u32_e64 v2, null, s29, v5, vcc_lo              // 000000003988: D5207C02 01AA0A1D
	s_mul_f32 s2, s0, 0x5f7ffffc                               // 000000003990: A202FF00 5F7FFFFC
	v_add_co_u32 v3, vcc_lo, s18, v6                           // 000000003998: D7006A03 00020C12
	v_rcp_iflag_f32_e32 v6, v13                                // 0000000039A0: 7E0C570D
	s_wait_alu 0xfffe                                          // 0000000039A4: BF88FFFE
	s_mul_f32 s3, s2, 0x2f800000                               // 0000000039A8: A203FF02 2F800000
	v_mov_b32_e32 v13, 0                                       // 0000000039B0: 7E1A0280
	s_wait_alu 0xfffd                                          // 0000000039B4: BF88FFFD
	v_add_co_ci_u32_e64 v10, null, s19, v7, vcc_lo             // 0000000039B8: D5207C0A 01AA0E13
	s_wait_alu 0xfffe                                          // 0000000039C0: BF88FFFE
	s_trunc_f32 s5, s3                                         // 0000000039C4: BE856203
	v_add_co_u32 v11, vcc_lo, s12, v11                         // 0000000039C8: D7006A0B 0002160C
	s_wait_alu 0xfffd                                          // 0000000039D0: BF88FFFD
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo            // 0000000039D4: D5207C0C 01AA180D
	v_mul_f32_e32 v6, 0x4f7ffffe, v6                           // 0000000039DC: 100C0CFF 4F7FFFFE
	s_wait_alu 0xfffe                                          // 0000000039E4: BF88FFFE
	s_fmamk_f32 s2, s5, 0xcf800000, s2                         // 0000000039E8: A3020205 CF800000
	s_cvt_u32_f32 s5, s5                                       // 0000000039F0: BE856705
	s_mov_b64 s[0:1], 0                                        // 0000000039F4: BE800180
	s_mov_b32 s3, 0                                            // 0000000039F8: BE830080
	v_cvt_u32_f32_e32 v15, v6                                  // 0000000039FC: 7E1E0F06
	s_wait_alu 0xfffe                                          // 000000003A00: BF88FFFE
	s_cvt_u32_f32 s4, s2                                       // 000000003A04: BE846702
	s_sub_co_i32 s35, 0, s24                                   // 000000003A08: 81A31880
	s_or_b64 s[8:9], s[0:1], s[24:25]                          // 000000003A0C: 8C881800
	s_mov_b32 s8, s3                                           // 000000003A10: BE880003
	s_mov_b32 s2, -1                                           // 000000003A14: BE8200C1
	s_wait_alu 0xfffe                                          // 000000003A18: BF88FFFE
	s_cmp_lg_u64 s[8:9], 0                                     // 000000003A1C: BF118008
	s_cbranch_scc0 104                                         // 000000003A20: BFA10068 <ullm_paged_decode_attn_f32_kernel+0x11c4>
	s_mul_u64 s[8:9], s[6:7], s[4:5]                           // 000000003A24: AA880406
	s_wait_alu 0xfffe                                          // 000000003A28: BF88FFFE
	s_mul_hi_u32 s31, s4, s9                                   // 000000003A2C: 969F0904
	s_mul_i32 s30, s4, s9                                      // 000000003A30: 961E0904
	s_mul_hi_u32 s2, s4, s8                                    // 000000003A34: 96820804
	s_mul_hi_u32 s44, s5, s9                                   // 000000003A38: 96AC0905
	s_wait_alu 0xfffe                                          // 000000003A3C: BF88FFFE
	s_add_nc_u64 s[30:31], s[2:3], s[30:31]                    // 000000003A40: A99E1E02
	s_mul_hi_u32 s2, s5, s8                                    // 000000003A44: 96820805
	s_mul_i32 s8, s5, s8                                       // 000000003A48: 96080805
	s_wait_alu 0xfffe                                          // 000000003A4C: BF88FFFE
	s_add_co_u32 s8, s30, s8                                   // 000000003A50: 8008081E
	s_add_co_ci_u32 s2, s31, s2                                // 000000003A54: 8202021F
	s_add_co_ci_u32 s31, s44, 0                                // 000000003A58: 821F802C
	s_mul_i32 s30, s5, s9                                      // 000000003A5C: 961E0905
	s_wait_alu 0xfffe                                          // 000000003A60: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[30:31]                      // 000000003A64: A9881E02
	s_wait_alu 0xfffe                                          // 000000003A68: BF88FFFE
	s_add_co_u32 s8, s4, s8                                    // 000000003A6C: 80080804
	s_cselect_b32 s2, -1, 0                                    // 000000003A70: 980280C1
	s_wait_alu 0xfffe                                          // 000000003A74: BF88FFFE
	s_cmp_lg_u32 s2, 0                                         // 000000003A78: BF078002
	s_add_co_ci_u32 s9, s5, s9                                 // 000000003A7C: 82090905
	s_wait_alu 0xfffe                                          // 000000003A80: BF88FFFE
	s_mul_u64 s[30:31], s[6:7], s[8:9]                         // 000000003A84: AA9E0806
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003A88: BF870009
	s_mul_hi_u32 s45, s8, s31                                  // 000000003A8C: 96AD1F08
	s_mul_i32 s44, s8, s31                                     // 000000003A90: 962C1F08
	s_mul_hi_u32 s2, s8, s30                                   // 000000003A94: 96821E08
	s_mul_hi_u32 s46, s9, s30                                  // 000000003A98: 96AE1E09
	s_mul_i32 s30, s9, s30                                     // 000000003A9C: 961E1E09
	s_wait_alu 0xfffe                                          // 000000003AA0: BF88FFFE
	s_add_nc_u64 s[44:45], s[2:3], s[44:45]                    // 000000003AA4: A9AC2C02
	s_mul_hi_u32 s47, s9, s31                                  // 000000003AA8: 96AF1F09
	s_add_co_u32 s2, s44, s30                                  // 000000003AAC: 80021E2C
	s_add_co_ci_u32 s2, s45, s46                               // 000000003AB0: 82022E2D
	s_mul_i32 s30, s9, s31                                     // 000000003AB4: 961E1F09
	s_wait_alu 0xfffe                                          // 000000003AB8: BF88FFFE
	s_add_co_ci_u32 s31, s47, 0                                // 000000003ABC: 821F802F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003AC0: BF870499
	s_add_nc_u64 s[30:31], s[2:3], s[30:31]                    // 000000003AC4: A99E1E02
	s_add_co_u32 s8, s8, s30                                   // 000000003AC8: 80081E08
	s_cselect_b32 s30, -1, 0                                   // 000000003ACC: 981E80C1
	s_wait_alu 0xfffe                                          // 000000003AD0: BF88FFFE
	s_mul_hi_u32 s2, s0, s8                                    // 000000003AD4: 96820800
	s_cmp_lg_u32 s30, 0                                        // 000000003AD8: BF07801E
	s_mul_hi_u32 s44, s1, s8                                   // 000000003ADC: 96AC0801
	s_add_co_ci_u32 s30, s9, s31                               // 000000003AE0: 821E1F09
	s_mul_i32 s31, s1, s8                                      // 000000003AE4: 961F0801
	s_mul_hi_u32 s9, s0, s30                                   // 000000003AE8: 96891E00
	s_mul_i32 s8, s0, s30                                      // 000000003AEC: 96081E00
	s_mul_hi_u32 s45, s1, s30                                  // 000000003AF0: 96AD1E01
	s_wait_alu 0xfffe                                          // 000000003AF4: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[8:9]                        // 000000003AF8: A9880802
	s_mul_i32 s30, s1, s30                                     // 000000003AFC: 961E1E01
	s_wait_alu 0xfffe                                          // 000000003B00: BF88FFFE
	s_add_co_u32 s2, s8, s31                                   // 000000003B04: 80021F08
	s_add_co_ci_u32 s2, s9, s44                                // 000000003B08: 82022C09
	s_add_co_ci_u32 s31, s45, 0                                // 000000003B0C: 821F802D
	s_wait_alu 0xfffe                                          // 000000003B10: BF88FFFE
	s_add_nc_u64 s[8:9], s[2:3], s[30:31]                      // 000000003B14: A9881E02
	s_wait_alu 0xfffe                                          // 000000003B18: BF88FFFE
	s_mul_u64 s[30:31], s[24:25], s[8:9]                       // 000000003B1C: AA9E0818
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003B20: BF870009
	s_sub_co_u32 s2, s0, s30                                   // 000000003B24: 80821E00
	s_cselect_b32 s30, -1, 0                                   // 000000003B28: 981E80C1
	s_sub_co_i32 s44, s1, s31                                  // 000000003B2C: 81AC1F01
	s_cmp_lg_u32 s30, 0                                        // 000000003B30: BF07801E
	s_sub_co_ci_u32 s44, s44, s25                              // 000000003B34: 82AC192C
	s_wait_alu 0xfffe                                          // 000000003B38: BF88FFFE
	s_sub_co_u32 s45, s2, s24                                  // 000000003B3C: 80AD1802
	s_cselect_b32 s46, -1, 0                                   // 000000003B40: 982E80C1
	s_wait_alu 0xfffe                                          // 000000003B44: BF88FFFE
	s_cmp_lg_u32 s46, 0                                        // 000000003B48: BF07802E
	s_sub_co_ci_u32 s44, s44, 0                                // 000000003B4C: 82AC802C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003B50: BF870009
	s_cmp_ge_u32 s44, s25                                      // 000000003B54: BF09192C
	s_cselect_b32 s46, -1, 0                                   // 000000003B58: 982E80C1
	s_cmp_ge_u32 s45, s24                                      // 000000003B5C: BF09182D
	s_cselect_b32 s47, -1, 0                                   // 000000003B60: 982F80C1
	s_cmp_eq_u32 s44, s25                                      // 000000003B64: BF06192C
	s_add_nc_u64 s[44:45], s[8:9], 1                           // 000000003B68: A9AC8108
	s_wait_alu 0xfffe                                          // 000000003B6C: BF88FFFE
	s_cselect_b32 s48, s47, s46                                // 000000003B70: 98302E2F
	s_add_nc_u64 s[46:47], s[8:9], 2                           // 000000003B74: A9AE8208
	s_cmp_lg_u32 s48, 0                                        // 000000003B78: BF078030
	s_wait_alu 0xfffe                                          // 000000003B7C: BF88FFFE
	s_cselect_b32 s44, s46, s44                                // 000000003B80: 982C2C2E
	s_cselect_b32 s45, s47, s45                                // 000000003B84: 982D2D2F
	s_cmp_lg_u32 s30, 0                                        // 000000003B88: BF07801E
	s_sub_co_ci_u32 s30, s1, s31                               // 000000003B8C: 829E1F01
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003B90: BF870009
	s_cmp_ge_u32 s30, s25                                      // 000000003B94: BF09191E
	s_cselect_b32 s31, -1, 0                                   // 000000003B98: 981F80C1
	s_cmp_ge_u32 s2, s24                                       // 000000003B9C: BF091802
	s_cselect_b32 s2, -1, 0                                    // 000000003BA0: 980280C1
	s_cmp_eq_u32 s30, s25                                      // 000000003BA4: BF06191E
	s_wait_alu 0xfffe                                          // 000000003BA8: BF88FFFE
	s_cselect_b32 s2, s2, s31                                  // 000000003BAC: 98021F02
	s_wait_alu 0xfffe                                          // 000000003BB0: BF88FFFE
	s_cmp_lg_u32 s2, 0                                         // 000000003BB4: BF078002
	s_mov_b32 s2, 0                                            // 000000003BB8: BE820080
	s_cselect_b32 s9, s45, s9                                  // 000000003BBC: 9809092D
	s_cselect_b32 s8, s44, s8                                  // 000000003BC0: 9808082C
	s_wait_alu 0xfffe                                          // 000000003BC4: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000003BC8: 916A027E
	s_wait_alu 0xfffe                                          // 000000003BCC: BF88FFFE
	s_cbranch_vccnz 25                                         // 000000003BD0: BFA40019 <ullm_paged_decode_attn_f32_kernel+0x1238>
	v_readfirstlane_b32 s2, v15                                // 000000003BD4: 7E04050F
	s_mul_i32 s8, s35, s2                                      // 000000003BD8: 96080223
	s_wait_alu 0xfffe                                          // 000000003BDC: BF88FFFE
	s_mul_hi_u32 s8, s2, s8                                    // 000000003BE0: 96880802
	s_wait_alu 0xfffe                                          // 000000003BE4: BF88FFFE
	s_add_co_i32 s2, s2, s8                                    // 000000003BE8: 81020802
	s_wait_alu 0xfffe                                          // 000000003BEC: BF88FFFE
	s_mul_hi_u32 s2, s0, s2                                    // 000000003BF0: 96820200
	s_wait_alu 0xfffe                                          // 000000003BF4: BF88FFFE
	s_mul_i32 s8, s2, s24                                      // 000000003BF8: 96081802
	s_add_co_i32 s9, s2, 1                                     // 000000003BFC: 81098102
	s_wait_alu 0xfffe                                          // 000000003C00: BF88FFFE
	s_sub_co_i32 s8, s0, s8                                    // 000000003C04: 81880800
	s_wait_alu 0xfffe                                          // 000000003C08: BF88FFFE
	s_sub_co_i32 s30, s8, s24                                  // 000000003C0C: 819E1808
	s_cmp_ge_u32 s8, s24                                       // 000000003C10: BF091808
	s_cselect_b32 s2, s9, s2                                   // 000000003C14: 98020209
	s_cselect_b32 s8, s30, s8                                  // 000000003C18: 9808081E
	s_wait_alu 0xfffe                                          // 000000003C1C: BF88FFFE
	s_add_co_i32 s9, s2, 1                                     // 000000003C20: 81098102
	s_cmp_ge_u32 s8, s24                                       // 000000003C24: BF091808
	s_wait_alu 0xfffe                                          // 000000003C28: BF88FFFE
	s_cselect_b32 s2, s9, s2                                   // 000000003C2C: 98020209
	s_wait_alu 0xfffe                                          // 000000003C30: BF88FFFE
	s_mov_b64 s[8:9], s[2:3]                                   // 000000003C34: BE880102
	s_wait_alu 0xfffe                                          // 000000003C38: BF88FFFE
	s_lshl_b64 s[30:31], s[8:9], 2                             // 000000003C3C: 849E8208
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003C40: BF870009
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]                  // 000000003C44: A99E1E14
	s_load_b32 s2, s[30:31], 0x0                               // 000000003C48: F400008F F8000000
	s_wait_kmcnt 0x0                                           // 000000003C50: BFC70000
	v_cmp_gt_u64_e64 s30, s[26:27], s[2:3]                     // 000000003C54: D45C001E 0000041A
	v_cmp_le_u64_e64 s44, s[26:27], s[2:3]                     // 000000003C5C: D45B002C 0000041A
	s_and_b32 vcc_lo, exec_lo, s30                             // 000000003C64: 8B6A1E7E
	s_mov_b32 s30, -1                                          // 000000003C68: BE9E00C1
	s_wait_alu 0xfffe                                          // 000000003C6C: BF88FFFE
	s_cbranch_vccz 116                                         // 000000003C70: BFA30074 <ullm_paged_decode_attn_f32_kernel+0x1444>
	s_sub_nc_u64 s[8:9], s[2:3], s[8:9]                        // 000000003C74: AA080802
	s_and_not1_b32 vcc_lo, exec_lo, s34                        // 000000003C78: 916A227E
	s_wait_alu 0xfffe                                          // 000000003C7C: BF88FFFE
	s_mul_u64 s[8:9], s[8:9], s[24:25]                         // 000000003C80: AA881808
	s_wait_alu 0xfffe                                          // 000000003C84: BF88FFFE
	s_add_nc_u64 s[8:9], s[8:9], s[0:1]                        // 000000003C88: A9880008
	s_wait_alu 0xfffe                                          // 000000003C8C: BF88FFFE
	v_mad_co_u64_u32 v[6:7], null, s8, s38, v[8:9]             // 000000003C90: D6FE7C06 04204C08
	s_mul_i32 s2, s9, s38                                      // 000000003C98: 96022609
	s_mul_i32 s8, s8, s39                                      // 000000003C9C: 96082708
	s_wait_alu 0xfffe                                          // 000000003CA0: BF88FFFE
	v_add3_u32 v7, s8, s2, v7                                  // 000000003CA4: D6550007 041C0408
	s_cbranch_vccnz 46                                         // 000000003CAC: BFA4002E <ullm_paged_decode_attn_f32_kernel+0x1368>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003CB0: BF870001
	v_mul_lo_u32 v20, v7, s40                                  // 000000003CB4: D72C0014 00005107
	v_mul_lo_u32 v21, v6, s41                                  // 000000003CBC: D72C0015 00005306
	v_mad_co_u64_u32 v[18:19], null, v6, s40, 0                // 000000003CC4: D6FE7C12 02005106
	s_mov_b64 s[8:9], 0                                        // 000000003CCC: BE880180
	s_mov_b64 s[30:31], s[40:41]                               // 000000003CD0: BE9E0128
	v_add3_u32 v19, v19, v21, v20                              // 000000003CD4: D6550013 04522B13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003CDC: BF870121
	v_lshlrev_b64_e32 v[19:20], 2, v[18:19]                    // 000000003CE0: 3E262482
	v_mov_b32_e32 v18, 0                                       // 000000003CE4: 7E240280
	v_add_co_u32 v19, vcc_lo, s16, v19                         // 000000003CE8: D7006A13 00022610
	s_wait_alu 0xfffd                                          // 000000003CF0: BF88FFFD
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003CF4: BF870003
	v_add_co_ci_u32_e64 v20, null, s17, v20, vcc_lo            // 000000003CF8: D5207C14 01AA2811
	s_wait_alu 0xfffe                                          // 000000003D00: BF88FFFE
	s_lshl_b64 s[46:47], s[8:9], 2                             // 000000003D04: 84AE8208
	s_add_nc_u64 s[30:31], s[30:31], -1                        // 000000003D08: A99EC11E
	s_wait_alu 0xfffe                                          // 000000003D0C: BF88FFFE
	v_add_co_u32 v21, vcc_lo, v11, s46                         // 000000003D10: D7006A15 00005D0B
	s_wait_alu 0xfffd                                          // 000000003D18: BF88FFFD
	v_add_co_ci_u32_e64 v22, null, s47, v12, vcc_lo            // 000000003D1C: D5207C16 01AA182F
	v_add_co_u32 v23, vcc_lo, v19, s46                         // 000000003D24: D7006A17 00005D13
	s_wait_alu 0xfffd                                          // 000000003D2C: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s47, v20, vcc_lo            // 000000003D30: D5207C18 01AA282F
	global_load_b32 v21, v[21:22], off                         // 000000003D38: EE05007C 00000015 00000015
	global_load_b32 v22, v[23:24], off                         // 000000003D44: EE05007C 00000016 00000017
	s_cmp_eq_u64 s[30:31], 0                                   // 000000003D50: BF10801E
	s_add_nc_u64 s[8:9], s[8:9], 1                             // 000000003D54: A9888108
	s_wait_loadcnt 0x0                                         // 000000003D58: BFC00000
	v_fmac_f32_e32 v18, v21, v22                               // 000000003D5C: 56242D15
	s_cbranch_scc0 65511                                       // 000000003D60: BFA1FFE7 <ullm_paged_decode_attn_f32_kernel+0x1300>
	s_branch 1                                                 // 000000003D64: BFA00001 <ullm_paged_decode_attn_f32_kernel+0x136c>
	v_mov_b32_e32 v18, 0                                       // 000000003D68: 7E240280
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000003D6C: BF8700D2
	v_mul_lo_u32 v19, v7, s42                                  // 000000003D70: D72C0013 00005507
	v_mul_lo_u32 v20, v6, s43                                  // 000000003D78: D72C0014 00005706
	v_mad_co_u64_u32 v[6:7], null, v6, s42, 0                  // 000000003D80: D6FE7C06 02005506
	s_mov_b32 s30, 0                                           // 000000003D88: BE9E0080
	v_add3_u32 v7, v7, v20, v19                                // 000000003D8C: D6550007 044E2907
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 000000003D94: 3E0C0C82
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003D98: BF870121
	v_add_co_u32 v6, vcc_lo, v3, v6                            // 000000003D9C: D7006A06 00020D03
	s_wait_alu 0xfffd                                          // 000000003DA4: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, v10, v7, vcc_lo              // 000000003DA8: D5207C07 01AA0F0A
	global_load_b32 v7, v[6:7], off                            // 000000003DB0: EE05007C 00000007 00000006
	v_fma_f32 v6, s33, v18, -v14                               // 000000003DBC: D6130006 843A2421
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003DC4: BF870121
	v_mul_f32_e32 v18, 0x3fb8aa3b, v6                          // 000000003DC8: 10240CFF 3FB8AA3B
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v6                   // 000000003DD0: 7C360CFF C2CE8ED0
	v_fma_f32 v19, 0x3fb8aa3b, v6, -v18                        // 000000003DD8: D6130013 844A0CFF 3FB8AA3B
	v_rndne_f32_e32 v20, v18                                   // 000000003DE4: 7E284712
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003DE8: BF870091
	v_dual_fmac_f32 v19, 0x32a5705f, v6 :: v_dual_sub_f32 v18, v18, v20// 000000003DEC: C80A0CFF 13122912 32A5705F
	v_add_f32_e32 v18, v18, v19                                // 000000003DF8: 06242712
	v_cvt_i32_f32_e32 v19, v20                                 // 000000003DFC: 7E261114
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003E00: BF870292
	v_exp_f32_e32 v18, v18                                     // 000000003E04: 7E244B12
	v_ldexp_f32 v18, v18, v19                                  // 000000003E08: D71C0012 00022712
	s_wait_alu 0xfffd                                          // 000000003E10: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000003E14: BF870131
	v_cndmask_b32_e32 v18, 0, v18, vcc_lo                      // 000000003E18: 02242480
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v6                   // 000000003E1C: 7C3C0CFF 42B17218
	s_wait_alu 0xfffd                                          // 000000003E24: BF88FFFD
	v_cndmask_b32_e32 v18, 0x7f800000, v18, vcc_lo             // 000000003E28: 022424FF 7F800000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003E30: BF870001
	v_add_f32_e32 v6, v17, v18                                 // 000000003E34: 060C2511
	s_wait_loadcnt 0x0                                         // 000000003E38: BFC00000
	v_fma_f32 v7, v18, v7, v16                                 // 000000003E3C: D6130007 04420F12
	s_and_b32 vcc_lo, exec_lo, s30                             // 000000003E44: 8B6A1E7E
	s_wait_alu 0xfffe                                          // 000000003E48: BF88FFFE
	s_cbranch_vccz 5                                           // 000000003E4C: BFA30005 <ullm_paged_decode_attn_f32_kernel+0x1464>
	v_dual_mov_b32 v6, v17 :: v_dual_mov_b32 v7, v16           // 000000003E50: CA100111 06060110
	global_store_b32 v[1:2], v13, off                          // 000000003E58: EE06807C 06800000 00000001
	s_add_nc_u64 s[0:1], s[0:1], 1                             // 000000003E64: A9808100
	s_wait_alu 0xfffe                                          // 000000003E68: BF88FFFE
	s_cmp_eq_u64 s[0:1], s[22:23]                              // 000000003E6C: BF101600
	s_cselect_b32 s2, -1, 0                                    // 000000003E70: 980280C1
	s_wait_alu 0xfffe                                          // 000000003E74: BF88FFFE
	s_or_b32 s2, s44, s2                                       // 000000003E78: 8C02022C
	s_wait_alu 0xfffe                                          // 000000003E7C: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000003E80: 916A027E
	s_wait_alu 0xfffe                                          // 000000003E84: BF88FFFE
	s_cbranch_vccz 6                                           // 000000003E88: BFA30006 <ullm_paged_decode_attn_f32_kernel+0x14a4>
	v_dual_mov_b32 v16, v7 :: v_dual_mov_b32 v17, v6           // 000000003E8C: CA100107 10100106
	s_branch 65245                                             // 000000003E94: BFA0FEDD <ullm_paged_decode_attn_f32_kernel+0x100c>
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0               // 000000003E98: CA100080 07060080
	s_mov_b32 s44, 0                                           // 000000003EA0: BEAC0080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003EA4: BF870009
	s_and_b32 vcc_lo, exec_lo, s44                             // 000000003EA8: 8B6A2C7E
	s_wait_alu 0xfffe                                          // 000000003EAC: BF88FFFE
	s_cbranch_vccnz 105                                        // 000000003EB0: BFA40069 <ullm_paged_decode_attn_f32_kernel+0x1658>
	v_div_scale_f32 v1, null, v6, v6, v7                       // 000000003EB4: D6FC7C01 041E0D06
	s_cmp_eq_u64 s[14:15], 0                                   // 000000003EBC: BF10800E
	v_rcp_f32_e32 v2, v1                                       // 000000003EC0: 7E045501
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003EC4: BF870095
	v_fma_f32 v3, -v1, v2, 1.0                                 // 000000003EC8: D6130003 23CA0501
	v_fmac_f32_e32 v2, v3, v2                                  // 000000003ED0: 56040503
	v_div_scale_f32 v3, vcc_lo, v7, v6, v7                     // 000000003ED4: D6FC6A03 041E0D07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003EDC: BF870091
	v_mul_f32_e32 v8, v3, v2                                   // 000000003EE0: 10100503
	v_fma_f32 v9, -v1, v8, v3                                  // 000000003EE4: D6130009 240E1101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003EEC: BF870091
	v_fmac_f32_e32 v8, v9, v2                                  // 000000003EF0: 56100509
	v_fma_f32 v1, -v1, v8, v3                                  // 000000003EF4: D6130001 240E1101
	s_wait_alu 0xfffd                                          // 000000003EFC: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003F00: BF870091
	v_div_fmas_f32 v1, v1, v2, v8                              // 000000003F04: D6370001 04220501
	v_div_fixup_f32 v1, v1, v6, v7                             // 000000003F0C: D6270001 041E0D01
	s_cbranch_scc1 856                                         // 000000003F14: BFA20358 <ullm_paged_decode_attn_f32_kernel+0x2278>
	v_add_co_u32 v2, vcc_lo, s14, v4                           // 000000003F18: D7006A02 0002080E
	s_wait_alu 0xfffd                                          // 000000003F20: BF88FFFD
	v_add_co_ci_u32_e64 v3, null, s15, v5, vcc_lo              // 000000003F24: D5207C03 01AA0A0F
	global_load_b32 v2, v[2:3], off                            // 000000003F2C: EE05007C 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000003F38: BFC00000
	v_mul_f32_e32 v3, 0xbfb8aa3b, v2                           // 000000003F3C: 100604FF BFB8AA3B
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2                   // 000000003F44: 7C3C04FF 42CE8ED0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003F4C: BF8700A2
	v_fma_f32 v6, 0xbfb8aa3b, v2, -v3                          // 000000003F50: D6130006 840E04FF BFB8AA3B
	v_rndne_f32_e32 v7, v3                                     // 000000003F5C: 7E0E4703
	v_dual_fmamk_f32 v6, v2, 0xb2a5705f, v6 :: v_dual_sub_f32 v3, v3, v7// 000000003F60: C88A0D02 06020F03 B2A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003F6C: BF870121
	v_add_f32_e32 v3, v3, v6                                   // 000000003F70: 06060D03
	v_cvt_i32_f32_e32 v6, v7                                   // 000000003F74: 7E0C1107
	v_exp_f32_e32 v3, v3                                       // 000000003F78: 7E064B03
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003F7C: BF8700A5
	v_ldexp_f32 v3, v3, v6                                     // 000000003F80: D71C0003 00020D03
	s_wait_alu 0xfffd                                          // 000000003F88: BF88FFFD
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000003F8C: 02060680
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2                   // 000000003F90: 7C3604FF C2B17218
	s_wait_alu 0xfffd                                          // 000000003F98: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003F9C: BF870092
	v_cndmask_b32_e32 v2, 0x7f800000, v3, vcc_lo               // 000000003FA0: 020406FF 7F800000
	v_add_f32_e32 v2, 1.0, v2                                  // 000000003FA8: 060404F2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003FAC: BF870121
	v_div_scale_f32 v3, null, v2, v2, 1.0                      // 000000003FB0: D6FC7C03 03CA0502
	v_div_scale_f32 v8, vcc_lo, 1.0, v2, 1.0                   // 000000003FB8: D6FC6A08 03CA04F2
	v_rcp_f32_e32 v6, v3                                       // 000000003FC0: 7E0C5503
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003FC4: BF870095
	v_fma_f32 v7, -v3, v6, 1.0                                 // 000000003FC8: D6130007 23CA0D03
	v_fmac_f32_e32 v6, v7, v6                                  // 000000003FD0: 560C0D07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003FD4: BF870091
	v_mul_f32_e32 v7, v8, v6                                   // 000000003FD8: 100E0D08
	v_fma_f32 v9, -v3, v7, v8                                  // 000000003FDC: D6130009 24220F03
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003FE4: BF870091
	v_fmac_f32_e32 v7, v9, v6                                  // 000000003FE8: 560E0D09
	v_fma_f32 v3, -v3, v7, v8                                  // 000000003FEC: D6130003 24220F03
	s_wait_alu 0xfffd                                          // 000000003FF4: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003FF8: BF870091
	v_div_fmas_f32 v3, v3, v6, v7                              // 000000003FFC: D6370003 041E0D03
	v_div_fixup_f32 v2, v3, v2, 1.0                            // 000000004004: D6270002 03CA0503
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000400C: BF870001
	v_mul_f32_e32 v6, v1, v2                                   // 000000004010: 100C0501
	v_add_co_u32 v2, vcc_lo, s28, v4                           // 000000004014: D7006A02 0002081C
	s_wait_alu 0xfffd                                          // 00000000401C: BF88FFFD
	v_add_co_ci_u32_e64 v3, null, s29, v5, vcc_lo              // 000000004020: D5207C03 01AA0A1D
	global_store_b32 v[2:3], v6, off                           // 000000004028: EE06807C 03000000 00000002
	s_cbranch_execnz 8                                         // 000000004034: BFA60008 <ullm_paged_decode_attn_f32_kernel+0x1658>
	v_add_co_u32 v2, vcc_lo, s28, v4                           // 000000004038: D7006A02 0002081C
	s_wait_alu 0xfffd                                          // 000000004040: BF88FFFD
	v_add_co_ci_u32_e64 v3, null, s29, v5, vcc_lo              // 000000004044: D5207C03 01AA0A1D
	global_store_b32 v[2:3], v1, off                           // 00000000404C: EE06807C 00800000 00000002
	s_or_b32 exec_lo, exec_lo, s11                             // 000000004058: 8C7E0B7E
	s_cbranch_execnz 64127                                     // 00000000405C: BFA6FA7F <ullm_paged_decode_attn_f32_kernel+0x5c>
	s_or_b64 s[0:1], s[36:37], s[38:39]                        // 000000004060: 8C802624
	s_mov_b32 s0, 0                                            // 000000004064: BE800080
	s_wait_alu 0xfffe                                          // 000000004068: BF88FFFE
	s_cmp_lg_u64 s[0:1], 0                                     // 00000000406C: BF118000
	s_cbranch_scc0 649                                         // 000000004070: BFA10289 <ullm_paged_decode_attn_f32_kernel+0x2098>
	s_cvt_f32_u32 s1, s38                                      // 000000004074: BE816526
	s_cvt_f32_u32 s2, s39                                      // 000000004078: BE826527
	s_sub_nc_u64 s[4:5], 0, s[38:39]                           // 00000000407C: AA042680
	s_mov_b32 s7, s0                                           // 000000004080: BE870000
	s_mov_b32 s31, s0                                          // 000000004084: BE9F0000
	s_wait_alu 0xfffe                                          // 000000004088: BF88FFFE
	s_fmamk_f32 s1, s2, 0x4f800000, s1                         // 00000000408C: A3010102 4F800000
	s_wait_alu 0xfffe                                          // 000000004094: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004098: BF87029A
	v_s_rcp_f32 s1, s1                                         // 00000000409C: D6840001 00000001
	s_mul_f32 s1, s1, 0x5f7ffffc                               // 0000000040A4: A201FF01 5F7FFFFC
	s_wait_alu 0xfffe                                          // 0000000040AC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000040B0: BF87052A
	s_mul_f32 s2, s1, 0x2f800000                               // 0000000040B4: A202FF01 2F800000
	s_wait_alu 0xfffe                                          // 0000000040BC: BF88FFFE
	s_trunc_f32 s2, s2                                         // 0000000040C0: BE826202
	s_wait_alu 0xfffe                                          // 0000000040C4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000040C8: BF8704BA
	s_fmamk_f32 s1, s2, 0xcf800000, s1                         // 0000000040CC: A3010102 CF800000
	s_cvt_u32_f32 s3, s2                                       // 0000000040D4: BE836702
	s_wait_alu 0xfffe                                          // 0000000040D8: BF88FFFE
	s_cvt_u32_f32 s2, s1                                       // 0000000040DC: BE826701
	s_wait_alu 0xfffe                                          // 0000000040E0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 0000000040E4: BF87000A
	s_mul_u64 s[8:9], s[4:5], s[2:3]                           // 0000000040E8: AA880204
	s_wait_alu 0xfffe                                          // 0000000040EC: BF88FFFE
	s_mul_hi_u32 s35, s2, s9                                   // 0000000040F0: 96A30902
	s_mul_i32 s34, s2, s9                                      // 0000000040F4: 96220902
	s_mul_hi_u32 s6, s2, s8                                    // 0000000040F8: 96860802
	s_mul_i32 s11, s3, s8                                      // 0000000040FC: 960B0803
	s_add_nc_u64 s[6:7], s[6:7], s[34:35]                      // 000000004100: A9862206
	s_mul_hi_u32 s1, s3, s8                                    // 000000004104: 96810803
	s_mul_hi_u32 s44, s3, s9                                   // 000000004108: 96AC0903
	s_wait_alu 0xfffe                                          // 00000000410C: BF88FFFE
	s_add_co_u32 s6, s6, s11                                   // 000000004110: 80060B06
	s_add_co_ci_u32 s30, s7, s1                                // 000000004114: 821E0107
	s_mul_i32 s8, s3, s9                                       // 000000004118: 96080903
	s_add_co_ci_u32 s9, s44, 0                                 // 00000000411C: 8209802C
	s_wait_alu 0xfffe                                          // 000000004120: BF88FFFE
	s_add_nc_u64 s[6:7], s[30:31], s[8:9]                      // 000000004124: A986081E
	s_mov_b32 s9, s0                                           // 000000004128: BE890000
	s_add_co_u32 s2, s2, s6                                    // 00000000412C: 80020602
	s_cselect_b32 s1, -1, 0                                    // 000000004130: 980180C1
	s_wait_alu 0xfffe                                          // 000000004134: BF88FFFE
	s_cmp_lg_u32 s1, 0                                         // 000000004138: BF078001
	s_add_co_ci_u32 s3, s3, s7                                 // 00000000413C: 82030703
	s_mov_b32 s7, s0                                           // 000000004140: BE870000
	s_wait_alu 0xfffe                                          // 000000004144: BF88FFFE
	s_mul_u64 s[4:5], s[4:5], s[2:3]                           // 000000004148: AA840204
	s_wait_alu 0xfffe                                          // 00000000414C: BF88FFFE
	s_mul_hi_u32 s31, s2, s5                                   // 000000004150: 969F0502
	s_mul_i32 s30, s2, s5                                      // 000000004154: 961E0502
	s_mul_hi_u32 s6, s2, s4                                    // 000000004158: 96860402
	s_mul_i32 s8, s3, s4                                       // 00000000415C: 96080403
	s_add_nc_u64 s[6:7], s[6:7], s[30:31]                      // 000000004160: A9861E06
	s_mul_hi_u32 s1, s3, s4                                    // 000000004164: 96810403
	s_mul_hi_u32 s11, s3, s5                                   // 000000004168: 968B0503
	s_mul_i32 s4, s3, s5                                       // 00000000416C: 96040503
	s_wait_alu 0xfffe                                          // 000000004170: BF88FFFE
	s_add_co_u32 s5, s6, s8                                    // 000000004174: 80050806
	s_add_co_ci_u32 s8, s7, s1                                 // 000000004178: 82080107
	s_add_co_ci_u32 s5, s11, 0                                 // 00000000417C: 8205800B
	s_mov_b32 s7, s0                                           // 000000004180: BE870000
	s_wait_alu 0xfffe                                          // 000000004184: BF88FFFE
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]                        // 000000004188: A9840408
	s_wait_alu 0xfffe                                          // 00000000418C: BF88FFFE
	s_add_co_u32 s1, s2, s4                                    // 000000004190: 80010402
	s_cselect_b32 s2, -1, 0                                    // 000000004194: 980280C1
	s_wait_alu 0xfffe                                          // 000000004198: BF88FFFE
	s_mul_hi_u32 s6, s36, s1                                   // 00000000419C: 96860124
	s_cmp_lg_u32 s2, 0                                         // 0000000041A0: BF078002
	s_mul_hi_u32 s8, s37, s1                                   // 0000000041A4: 96880125
	s_add_co_ci_u32 s4, s3, s5                                 // 0000000041A8: 82040503
	s_mul_i32 s1, s37, s1                                      // 0000000041AC: 96010125
	s_wait_alu 0xfffe                                          // 0000000041B0: BF88FFFE
	s_mul_hi_u32 s3, s36, s4                                   // 0000000041B4: 96830424
	s_mul_i32 s2, s36, s4                                      // 0000000041B8: 96020424
	s_mul_hi_u32 s5, s37, s4                                   // 0000000041BC: 96850425
	s_wait_alu 0xfffe                                          // 0000000041C0: BF88FFFE
	s_add_nc_u64 s[2:3], s[6:7], s[2:3]                        // 0000000041C4: A9820206
	s_mul_i32 s4, s37, s4                                      // 0000000041C8: 96040425
	s_wait_alu 0xfffe                                          // 0000000041CC: BF88FFFE
	s_add_co_u32 s1, s2, s1                                    // 0000000041D0: 80010102
	s_add_co_ci_u32 s8, s3, s8                                 // 0000000041D4: 82080803
	s_add_co_ci_u32 s5, s5, 0                                  // 0000000041D8: 82058005
	s_wait_alu 0xfffe                                          // 0000000041DC: BF88FFFE
	s_add_nc_u64 s[2:3], s[8:9], s[4:5]                        // 0000000041E0: A9820408
	s_wait_alu 0xfffe                                          // 0000000041E4: BF88FFFE
	s_mul_u64 s[4:5], s[38:39], s[2:3]                         // 0000000041E8: AA840226
	s_wait_alu 0xfffe                                          // 0000000041EC: BF88FFFE
	s_sub_co_u32 s1, s36, s4                                   // 0000000041F0: 80810424
	s_cselect_b32 s4, -1, 0                                    // 0000000041F4: 980480C1
	s_sub_co_i32 s6, s37, s5                                   // 0000000041F8: 81860525
	s_wait_alu 0xfffe                                          // 0000000041FC: BF88FFFE
	s_cmp_lg_u32 s4, 0                                         // 000000004200: BF078004
	s_sub_co_ci_u32 s6, s6, s39                                // 000000004204: 82862706
	s_sub_co_u32 s7, s1, s38                                   // 000000004208: 80872601
	s_cselect_b32 s8, -1, 0                                    // 00000000420C: 980880C1
	s_wait_alu 0xfffe                                          // 000000004210: BF88FFFE
	s_cmp_lg_u32 s8, 0                                         // 000000004214: BF078008
	s_sub_co_ci_u32 s6, s6, 0                                  // 000000004218: 82868006
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000421C: BF870009
	s_cmp_ge_u32 s6, s39                                       // 000000004220: BF092706
	s_cselect_b32 s8, -1, 0                                    // 000000004224: 980880C1
	s_cmp_ge_u32 s7, s38                                       // 000000004228: BF092607
	s_cselect_b32 s9, -1, 0                                    // 00000000422C: 980980C1
	s_cmp_eq_u32 s6, s39                                       // 000000004230: BF062706
	s_add_nc_u64 s[6:7], s[2:3], 1                             // 000000004234: A9868102
	s_wait_alu 0xfffe                                          // 000000004238: BF88FFFE
	s_cselect_b32 s11, s9, s8                                  // 00000000423C: 980B0809
	s_add_nc_u64 s[8:9], s[2:3], 2                             // 000000004240: A9888202
	s_wait_alu 0xfffe                                          // 000000004244: BF88FFFE
	s_cmp_lg_u32 s11, 0                                        // 000000004248: BF07800B
	s_cselect_b32 s6, s8, s6                                   // 00000000424C: 98060608
	s_cselect_b32 s7, s9, s7                                   // 000000004250: 98070709
	s_cmp_lg_u32 s4, 0                                         // 000000004254: BF078004
	s_sub_co_ci_u32 s4, s37, s5                                // 000000004258: 82840525
	s_wait_alu 0xfffe                                          // 00000000425C: BF88FFFE
	s_cmp_ge_u32 s4, s39                                       // 000000004260: BF092704
	s_cselect_b32 s5, -1, 0                                    // 000000004264: 980580C1
	s_cmp_ge_u32 s1, s38                                       // 000000004268: BF092601
	s_cselect_b32 s1, -1, 0                                    // 00000000426C: 980180C1
	s_cmp_eq_u32 s4, s39                                       // 000000004270: BF062704
	s_wait_alu 0xfffe                                          // 000000004274: BF88FFFE
	s_cselect_b32 s1, s1, s5                                   // 000000004278: 98010501
	s_wait_alu 0xfffe                                          // 00000000427C: BF88FFFE
	s_cmp_lg_u32 s1, 0                                         // 000000004280: BF078001
	s_cselect_b32 s45, s7, s3                                  // 000000004284: 982D0307
	s_cselect_b32 s44, s6, s2                                  // 000000004288: 982C0206
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000428C: 916A007E
	s_wait_alu 0xfffe                                          // 000000004290: BF88FFFE
	s_cbranch_vccnz 33                                         // 000000004294: BFA40021 <ullm_paged_decode_attn_f32_kernel+0x191c>
	v_cvt_f32_u32_e32 v1, s38                                  // 000000004298: 7E020C26
	s_sub_co_i32 s1, 0, s38                                    // 00000000429C: 81812680
	s_mov_b32 s45, 0                                           // 0000000042A0: BEAD0080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000042A4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 0000000042A8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000042AC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000042B4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000042B8: 7E020F01
	v_readfirstlane_b32 s0, v1                                 // 0000000042BC: 7E000501
	s_wait_alu 0xfffe                                          // 0000000042C0: BF88FFFE
	s_mul_i32 s1, s1, s0                                       // 0000000042C4: 96010001
	s_wait_alu 0xfffe                                          // 0000000042C8: BF88FFFE
	s_mul_hi_u32 s1, s0, s1                                    // 0000000042CC: 96810100
	s_wait_alu 0xfffe                                          // 0000000042D0: BF88FFFE
	s_add_co_i32 s0, s0, s1                                    // 0000000042D4: 81000100
	s_wait_alu 0xfffe                                          // 0000000042D8: BF88FFFE
	s_mul_hi_u32 s0, s36, s0                                   // 0000000042DC: 96800024
	s_wait_alu 0xfffe                                          // 0000000042E0: BF88FFFE
	s_mul_i32 s1, s0, s38                                      // 0000000042E4: 96012600
	s_add_co_i32 s2, s0, 1                                     // 0000000042E8: 81028100
	s_wait_alu 0xfffe                                          // 0000000042EC: BF88FFFE
	s_sub_co_i32 s1, s36, s1                                   // 0000000042F0: 81810124
	s_wait_alu 0xfffe                                          // 0000000042F4: BF88FFFE
	s_sub_co_i32 s3, s1, s38                                   // 0000000042F8: 81832601
	s_cmp_ge_u32 s1, s38                                       // 0000000042FC: BF092601
	s_cselect_b32 s0, s2, s0                                   // 000000004300: 98000002
	s_wait_alu 0xfffe                                          // 000000004304: BF88FFFE
	s_cselect_b32 s1, s3, s1                                   // 000000004308: 98010103
	s_add_co_i32 s2, s0, 1                                     // 00000000430C: 81028100
	s_wait_alu 0xfffe                                          // 000000004310: BF88FFFE
	s_cmp_ge_u32 s1, s38                                       // 000000004314: BF092601
	s_cselect_b32 s44, s2, s0                                  // 000000004318: 982C0002
	s_load_b32 s0, s[20:21], 0x0                               // 00000000431C: F400000A F8000000
	s_mov_b32 s11, 0                                           // 000000004324: BE8B0080
	v_mov_b32_e32 v1, 0                                        // 000000004328: 7E020280
	s_wait_alu 0xfffe                                          // 00000000432C: BF88FFFE
	s_mov_b32 s1, s11                                          // 000000004330: BE81000B
	s_wait_kmcnt 0x0                                           // 000000004334: BFC70000
	s_wait_alu 0xfffe                                          // 000000004338: BF88FFFE
	v_cmp_gt_u64_e64 s2, s[26:27], s[0:1]                      // 00000000433C: D45C0002 0000001A
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000004344: 8B6A027E
	s_mov_b32 s2, -1                                           // 000000004348: BE8200C1
	s_wait_alu 0xfffe                                          // 00000000434C: BF88FFFE
	s_cbranch_vccnz 17                                         // 000000004350: BFA40011 <ullm_paged_decode_attn_f32_kernel+0x1998>
	s_mov_b32 s2, exec_lo                                      // 000000004354: BE82007E
	v_cmpx_gt_u64_e64 s[42:43], v[0:1]                         // 000000004358: D4DC007E 0002002A
	s_cbranch_execz 10                                         // 000000004360: BFA5000A <ullm_paged_decode_attn_f32_kernel+0x198c>
	s_mul_u64 s[4:5], s[42:43], s[10:11]                       // 000000004364: AA840A2A
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v1, 2, v0       // 000000004368: CA220080 02000082
	s_wait_alu 0xfffe                                          // 000000004370: BF88FFFE
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000004374: 84848204
	s_wait_alu 0xfffe                                          // 000000004378: BF88FFFE
	s_add_nc_u64 s[4:5], s[28:29], s[4:5]                      // 00000000437C: A984041C
	global_store_b32 v1, v2, s[4:5]                            // 000000004380: EE068004 01000000 00000001
	s_wait_alu 0xfffe                                          // 00000000438C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004390: 8C7E027E
	s_mov_b32 s2, 0                                            // 000000004394: BE820080
	s_wait_alu 0xfffe                                          // 000000004398: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 00000000439C: 916A027E
	s_wait_alu 0xfffe                                          // 0000000043A0: BF88FFFE
	s_cbranch_vccnz 63917                                      // 0000000043A4: BFA4F9AD <ullm_paged_decode_attn_f32_kernel+0x5c>
	v_mov_b32_e32 v1, 0                                        // 0000000043A8: 7E020280
	s_cmp_lg_u64 s[22:23], 0                                   // 0000000043AC: BF118016
	s_mov_b64 s[30:31], 0                                      // 0000000043B0: BE9E0180
	s_cbranch_scc0 441                                         // 0000000043B4: BFA101B9 <ullm_paged_decode_attn_f32_kernel+0x209c>
	s_clz_i32_u32 s2, s45                                      // 0000000043B8: BE820A2D
	s_mul_u64 s[36:37], s[40:41], s[10:11]                     // 0000000043BC: AAA40A28
	s_wait_alu 0xfffe                                          // 0000000043C0: BF88FFFE
	s_min_u32 s4, s2, 32                                       // 0000000043C4: 8984A002
	s_lshl_b64 s[36:37], s[36:37], 2                           // 0000000043C8: 84A48224
	s_wait_alu 0xfffe                                          // 0000000043CC: BF88FFFE
	s_lshl_b64 s[2:3], s[44:45], s4                            // 0000000043D0: 8482042C
	s_add_nc_u64 s[12:13], s[12:13], s[36:37]                  // 0000000043D4: A98C240C
	s_wait_alu 0xfffe                                          // 0000000043D8: BF88FFFE
	s_min_u32 s2, s2, 1                                        // 0000000043DC: 89828102
	s_mul_u64 s[34:35], s[24:25], s[0:1]                       // 0000000043E0: AAA20018
	s_wait_alu 0xfffe                                          // 0000000043E4: BF88FFFE
	s_or_b32 s2, s3, s2                                        // 0000000043E8: 8C020203
	s_sub_co_i32 s3, 32, s4                                    // 0000000043EC: 818304A0
	s_wait_alu 0xfffe                                          // 0000000043F0: BF88FFFE
	s_cvt_f32_u32 s2, s2                                       // 0000000043F4: BE826502
	v_cmp_gt_u64_e64 s0, s[40:41], v[0:1]                      // 0000000043F8: D45C0000 00020028
	v_cmp_gt_u64_e64 s1, s[42:43], v[0:1]                      // 000000004400: D45C0001 0002002A
	v_cmp_gt_u32_e64 s4, 32, v0                                // 000000004408: D44C0004 000200A0
	s_wait_alu 0xfffe                                          // 000000004410: BF88FFFE
	v_ldexp_f32 v2, s2, s3                                     // 000000004414: D71C0002 00000602
	s_sub_co_i32 s2, 0, s44                                    // 00000000441C: 81822C80
	v_cmp_gt_u32_e64 s3, 64, v0                                // 000000004420: D44C0003 000200C0
	v_cmp_gt_u32_e64 s5, 16, v0                                // 000000004428: D44C0005 00020090
	v_cmp_gt_u32_e64 s6, 8, v0                                 // 000000004430: D44C0006 00020088
	v_rcp_f32_e32 v2, v2                                       // 000000004438: 7E045502
	v_cmp_gt_u32_e64 s7, 4, v0                                 // 00000000443C: D44C0007 00020084
	v_cmp_gt_u32_e64 s8, 2, v0                                 // 000000004444: D44C0008 00020082
	v_cmp_eq_u32_e64 s9, 0, v0                                 // 00000000444C: D44A0009 00020080
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004454: BF870095
	v_mul_f32_e32 v2, 0x4f7ffffe, v2                           // 000000004458: 100404FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v2                                   // 000000004460: 7E040F02
	s_wait_alu 0xfffe                                          // 000000004464: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004468: BF870121
	v_mul_lo_u32 v3, s2, v2                                    // 00000000446C: D72C0003 00020402
	v_cmp_gt_u32_e64 s2, 0x80, v0                              // 000000004474: D44C0002 000200FF 00000080
	v_mul_hi_u32 v3, v2, v3                                    // 000000004480: D72D0003 00020702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004488: BF870091
	v_add_nc_u32_e32 v2, v2, v3                                // 00000000448C: 4A040702
	v_mul_hi_u32 v2, v2, s10                                   // 000000004490: D72D0002 00001502
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004498: BF8700A1
	v_dual_mov_b32 v12, 0xff7fffff :: v_dual_add_nc_u32 v5, 1, v2// 00000000449C: CA2000FF 0C040481 FF7FFFFF
	v_mul_lo_u32 v3, v2, s44                                   // 0000000044A8: D72C0003 00005902
	v_sub_nc_u32_e32 v3, s10, v3                               // 0000000044B0: 4C06060A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000044B4: BF8701C1
	v_subrev_nc_u32_e32 v11, s44, v3                           // 0000000044B8: 4E16062C
	v_cmp_le_u32_e32 vcc_lo, s44, v3                           // 0000000044BC: 7C96062C
	s_wait_alu 0xfffd                                          // 0000000044C0: BF88FFFD
	v_dual_cndmask_b32 v5, v2, v5 :: v_dual_lshlrev_b32 v6, 2, v0// 0000000044C4: CA620B02 05060082
	v_dual_cndmask_b32 v11, v3, v11 :: v_dual_mov_b32 v4, v1   // 0000000044CC: CA501703 0B040101
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000044D4: BF870192
	v_add_co_u32 v7, s16, s16, v6                              // 0000000044D8: D7001007 00020C10
	v_add_nc_u32_e32 v13, 1, v5                                // 0000000044E0: 4A1A0A81
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000044E4: BF870003
	v_cmp_le_u32_e32 vcc_lo, s44, v11                          // 0000000044E8: 7C96162C
	s_wait_alu 0xf1ff                                          // 0000000044EC: BF88F1FF
	v_add_co_ci_u32_e64 v8, null, s17, 0, s16                  // 0000000044F0: D5207C08 00410011
	v_add_co_u32 v9, s16, s18, v6                              // 0000000044F8: D7001009 00020C12
	v_add_co_u32 v2, s12, s12, v6                              // 000000004500: D7000C02 00020C0C
	s_wait_alu 0xf1ff                                          // 000000004508: BF88F1FF
	v_add_co_ci_u32_e64 v10, null, s19, 0, s16                 // 00000000450C: D5207C0A 00410013
	v_add_co_ci_u32_e64 v3, null, s13, 0, s12                  // 000000004514: D5207C03 0031000D
	s_wait_alu 0xfffd                                          // 00000000451C: BF88FFFD
	v_cndmask_b32_e32 v11, v5, v13, vcc_lo                     // 000000004520: 02161B05
	v_mov_b32_e32 v5, v1                                       // 000000004524: 7E0A0301
	s_mov_b32 s13, 0                                           // 000000004528: BE8D0080
	s_mov_b64 s[16:17], s[24:25]                               // 00000000452C: BE900118
	s_mov_b64 s[18:19], 0                                      // 000000004530: BE920180
	s_branch 14                                                // 000000004534: BFA0000E <ullm_paged_decode_attn_f32_kernel+0x1b70>
	s_wait_alu 0xfffe                                          // 000000004538: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s36                             // 00000000453C: 8C7E247E
	s_wait_alu 0xfffe                                          // 000000004540: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004544: 8C7E0C7E
	s_add_nc_u64 s[18:19], s[18:19], 1                         // 000000004548: A9928112
	s_add_nc_u64 s[34:35], s[34:35], 1                         // 00000000454C: A9A28122
	s_wait_alu 0xfffe                                          // 000000004550: BF88FFFE
	s_cmp_eq_u64 s[22:23], s[18:19]                            // 000000004554: BF101216
	s_mov_b32 s36, 0                                           // 000000004558: BEA40080
	s_cselect_b32 s12, -1, 0                                   // 00000000455C: 980C80C1
	s_wait_alu 0xfffe                                          // 000000004560: BF88FFFE
	s_and_b32 vcc_lo, exec_lo, s12                             // 000000004564: 8B6A0C7E
	s_wait_alu 0xfffe                                          // 000000004568: BF88FFFE
	s_cbranch_vccnz 310                                        // 00000000456C: BFA40136 <ullm_paged_decode_attn_f32_kernel+0x2048>
	s_wait_alu 0xfffe                                          // 000000004570: BF88FFFE
	s_cmp_lg_u64 s[18:19], s[16:17]                            // 000000004574: BF111012
	s_mov_b32 s36, -1                                          // 000000004578: BEA400C1
	s_cbranch_scc1 19                                          // 00000000457C: BFA20013 <ullm_paged_decode_attn_f32_kernel+0x1bcc>
	s_lshl_b64 s[34:35], s[30:31], 2                           // 000000004580: 84A2821E
	s_mov_b32 s37, 0                                           // 000000004584: BEA50080
	s_add_nc_u64 s[34:35], s[20:21], s[34:35]                  // 000000004588: A9A22214
	s_wait_loadcnt 0x0                                         // 00000000458C: BFC00000
	global_load_b32 v13, v1, s[34:35] offset:4                 // 000000004590: EE050022 0000000D 00000401
	s_wait_loadcnt 0x0                                         // 00000000459C: BFC00000
	v_readfirstlane_b32 s12, v13                               // 0000000045A0: 7E18050D
	s_wait_alu 0xf1ff                                          // 0000000045A4: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000045A8: BF870001
	v_cmp_gt_u64_e64 s34, s[26:27], s[12:13]                   // 0000000045AC: D45C0022 0000181A
	s_and_b32 vcc_lo, exec_lo, s34                             // 0000000045B4: 8B6A227E
	s_wait_alu 0xfffe                                          // 0000000045B8: BF88FFFE
	s_cbranch_vccz 4                                           // 0000000045BC: BFA30004 <ullm_paged_decode_attn_f32_kernel+0x1bd0>
	s_add_nc_u64 s[30:31], s[30:31], 1                         // 0000000045C0: A99E811E
	s_add_nc_u64 s[16:17], s[16:17], s[24:25]                  // 0000000045C4: A9901810
	s_mul_u64 s[34:35], s[24:25], s[12:13]                     // 0000000045C8: AAA20C18
	s_mov_b32 s37, -1                                          // 0000000045CC: BEA500C1
	s_wait_alu 0xfffe                                          // 0000000045D0: BF88FFFE
	s_and_b32 vcc_lo, exec_lo, s37                             // 0000000045D4: 8B6A257E
	s_wait_alu 0xfffe                                          // 0000000045D8: BF88FFFE
	s_cbranch_vccz 281                                         // 0000000045DC: BFA30119 <ullm_paged_decode_attn_f32_kernel+0x2044>
	s_mul_u64 s[36:37], s[34:35], s[38:39]                     // 0000000045E0: AAA42622
	v_mov_b32_e32 v15, 0                                       // 0000000045E4: 7E1E0280
	s_wait_loadcnt 0x0                                         // 0000000045E8: BFC00000
	s_wait_alu 0xfffe                                          // 0000000045EC: BF88FFFE
	v_add_co_u32 v13, s12, s36, v11                            // 0000000045F0: D7000C0D 00021624
	s_wait_alu 0xf1ff                                          // 0000000045F8: BF88F1FF
	v_add_co_ci_u32_e64 v14, null, s37, 0, s12                 // 0000000045FC: D5207C0E 00310025
	s_and_saveexec_b32 s12, s0                                 // 000000004604: BE8C2000
	s_cbranch_execz 27                                         // 000000004608: BFA5001B <ullm_paged_decode_attn_f32_kernel+0x1c78>
	v_mul_lo_u32 v15, s40, v14                                 // 00000000460C: D72C000F 00021C28
	v_mul_lo_u32 v16, s41, v13                                 // 000000004614: D72C0010 00021A29
	v_mul_hi_u32 v17, s40, v13                                 // 00000000461C: D72D0011 00021A28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004624: BF870122
	v_add_nc_u32_e32 v16, v15, v16                             // 000000004628: 4A20210F
	v_mul_lo_u32 v15, s40, v13                                 // 00000000462C: D72C000F 00021A28
	v_add_nc_u32_e32 v16, v16, v17                             // 000000004634: 4A202310
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004638: BF870091
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]                    // 00000000463C: 3E1E1E82
	v_add_co_u32 v15, vcc_lo, v7, v15                          // 000000004640: D7006A0F 00021F07
	s_wait_alu 0xfffd                                          // 000000004648: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000464C: BF870002
	v_add_co_ci_u32_e64 v16, null, v8, v16, vcc_lo             // 000000004650: D5207C10 01AA2108
	global_load_b32 v17, v[2:3], off                           // 000000004658: EE05007C 00000011 00000002
	global_load_b32 v15, v[15:16], off                         // 000000004664: EE05007C 0000000F 0000000F
	s_wait_loadcnt 0x0                                         // 000000004670: BFC00000
	v_mul_f32_e32 v15, v17, v15                                // 000000004674: 101E1F11
	s_wait_alu 0xfffe                                          // 000000004678: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 00000000467C: 8C7E0C7E
	ds_store_b32 v6, v15                                       // 000000004680: D8340000 00000F06
	s_wait_storecnt_dscnt 0x0                                  // 000000004688: BFC90000
	s_barrier_signal -1                                        // 00000000468C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004690: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004694: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s2                                 // 0000000046A0: BE8C2002
	s_cbranch_execz 6                                          // 0000000046A4: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1cc0>
	ds_load_2addr_stride64_b32 v[15:16], v6 offset1:2          // 0000000046A8: D8E00200 0F000006
	s_wait_dscnt 0x0                                           // 0000000046B0: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 0000000046B4: 061E1F10
	ds_store_b32 v6, v15                                       // 0000000046B8: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 0000000046C0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 0000000046C4: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000046C8: BFC80000
	s_barrier_signal -1                                        // 0000000046CC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000046D0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000046D4: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s3                                 // 0000000046E0: BE8C2003
	s_cbranch_execz 6                                          // 0000000046E4: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1d00>
	ds_load_2addr_stride64_b32 v[15:16], v6 offset1:1          // 0000000046E8: D8E00100 0F000006
	s_wait_dscnt 0x0                                           // 0000000046F0: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 0000000046F4: 061E1F10
	ds_store_b32 v6, v15                                       // 0000000046F8: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004700: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004704: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004708: BFC80000
	s_barrier_signal -1                                        // 00000000470C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004710: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004714: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s4                                 // 000000004720: BE8C2004
	s_cbranch_execz 6                                          // 000000004724: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1d40>
	ds_load_2addr_b32 v[15:16], v6 offset1:32                  // 000000004728: D8DC2000 0F000006
	s_wait_dscnt 0x0                                           // 000000004730: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 000000004734: 061E1F10
	ds_store_b32 v6, v15                                       // 000000004738: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004740: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004744: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004748: BFC80000
	s_barrier_signal -1                                        // 00000000474C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004750: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004754: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s5                                 // 000000004760: BE8C2005
	s_cbranch_execz 6                                          // 000000004764: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1d80>
	ds_load_2addr_b32 v[15:16], v6 offset1:16                  // 000000004768: D8DC1000 0F000006
	s_wait_dscnt 0x0                                           // 000000004770: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 000000004774: 061E1F10
	ds_store_b32 v6, v15                                       // 000000004778: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004780: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004784: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004788: BFC80000
	s_barrier_signal -1                                        // 00000000478C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004790: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004794: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s6                                 // 0000000047A0: BE8C2006
	s_cbranch_execz 6                                          // 0000000047A4: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1dc0>
	ds_load_2addr_b32 v[15:16], v6 offset1:8                   // 0000000047A8: D8DC0800 0F000006
	s_wait_dscnt 0x0                                           // 0000000047B0: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 0000000047B4: 061E1F10
	ds_store_b32 v6, v15                                       // 0000000047B8: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 0000000047C0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 0000000047C4: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000047C8: BFC80000
	s_barrier_signal -1                                        // 0000000047CC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000047D0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000047D4: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s7                                 // 0000000047E0: BE8C2007
	s_cbranch_execz 6                                          // 0000000047E4: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1e00>
	ds_load_2addr_b32 v[15:16], v6 offset1:4                   // 0000000047E8: D8DC0400 0F000006
	s_wait_dscnt 0x0                                           // 0000000047F0: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 0000000047F4: 061E1F10
	ds_store_b32 v6, v15                                       // 0000000047F8: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004800: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004804: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004808: BFC80000
	s_barrier_signal -1                                        // 00000000480C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004810: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004814: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s8                                 // 000000004820: BE8C2008
	s_cbranch_execz 6                                          // 000000004824: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1e40>
	ds_load_2addr_b32 v[15:16], v6 offset1:2                   // 000000004828: D8DC0200 0F000006
	s_wait_dscnt 0x0                                           // 000000004830: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 000000004834: 061E1F10
	ds_store_b32 v6, v15                                       // 000000004838: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004840: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004844: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004848: BFC80000
	s_barrier_signal -1                                        // 00000000484C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004850: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004854: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s9                                 // 000000004860: BE8C2009
	s_cbranch_execz 6                                          // 000000004864: BFA50006 <ullm_paged_decode_attn_f32_kernel+0x1e80>
	ds_load_2addr_b32 v[15:16], v6 offset1:1                   // 000000004868: D8DC0100 0F000006
	s_wait_dscnt 0x0                                           // 000000004870: BFC60000
	v_add_f32_e32 v15, v16, v15                                // 000000004874: 061E1F10
	ds_store_b32 v6, v15                                       // 000000004878: D8340000 00000F06
	s_wait_alu 0xfffe                                          // 000000004880: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004884: 8C7E0C7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004888: BFC80000
	s_barrier_signal -1                                        // 00000000488C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004890: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004894: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s12, s1                                 // 0000000048A0: BE8C2001
	s_cbranch_execz 65318                                      // 0000000048A4: BFA5FF26 <ullm_paged_decode_attn_f32_kernel+0x1b40>
	v_mul_lo_u32 v14, s42, v14                                 // 0000000048A8: D72C000E 00021C2A
	v_mul_lo_u32 v15, s43, v13                                 // 0000000048B0: D72C000F 00021A2B
	v_mul_hi_u32 v16, s42, v13                                 // 0000000048B8: D72D0010 00021A2A
	v_mul_lo_u32 v13, s42, v13                                 // 0000000048C0: D72C000D 00021A2A
	s_mov_b32 s36, exec_lo                                     // 0000000048C8: BEA4007E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000048CC: BF870093
	v_add_nc_u32_e32 v14, v14, v15                             // 0000000048D0: 4A1C1F0E
	v_add_nc_u32_e32 v14, v14, v16                             // 0000000048D4: 4A1C210E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000048D8: BF870091
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]                    // 0000000048DC: 3E1A1A82
	v_add_co_u32 v13, vcc_lo, v9, v13                          // 0000000048E0: D7006A0D 00021B09
	s_wait_alu 0xfffd                                          // 0000000048E8: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000048EC: BF8700D2
	v_add_co_ci_u32_e64 v14, null, v10, v14, vcc_lo            // 0000000048F0: D5207C0E 01AA1D0A
	global_load_b32 v13, v[13:14], off                         // 0000000048F8: EE05007C 0000000D 0000000D
	ds_load_b32 v14, v1                                        // 000000004904: D8D80000 0E000001
	s_wait_dscnt 0x0                                           // 00000000490C: BFC60000
	v_mul_f32_e32 v14, s33, v14                                // 000000004910: 101C1C21
	v_cmpx_ngt_f32_e32 v14, v12                                // 000000004914: 7D36190E
	s_wait_alu 0xfffe                                          // 000000004918: BF88FFFE
	s_xor_b32 s36, exec_lo, s36                                // 00000000491C: 8D24247E
	s_cbranch_execz 32                                         // 000000004920: BFA50020 <ullm_paged_decode_attn_f32_kernel+0x1fa4>
	v_sub_f32_e32 v14, v14, v12                                // 000000004924: 081C190E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004928: BF870091
	v_mul_f32_e32 v15, 0x3fb8aa3b, v14                         // 00000000492C: 101E1CFF 3FB8AA3B
	v_fma_f32 v16, 0x3fb8aa3b, v14, -v15                       // 000000004934: D6130010 843E1CFF 3FB8AA3B
	v_rndne_f32_e32 v17, v15                                   // 000000004940: 7E22470F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000004944: BF870221
	v_sub_f32_e32 v15, v15, v17                                // 000000004948: 081E230F
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v14                  // 00000000494C: 7C361CFF C2CE8ED0
	v_fmac_f32_e32 v16, 0x32a5705f, v14                        // 000000004954: 56201CFF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000495C: BF870121
	v_add_f32_e32 v15, v15, v16                                // 000000004960: 061E210F
	v_cvt_i32_f32_e32 v16, v17                                 // 000000004964: 7E201111
	v_exp_f32_e32 v15, v15                                     // 000000004968: 7E1E4B0F
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000496C: BF8700A5
	v_ldexp_f32 v15, v15, v16                                  // 000000004970: D71C000F 0002210F
	s_wait_alu 0xfffd                                          // 000000004978: BF88FFFD
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo                      // 00000000497C: 021E1E80
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v14                  // 000000004980: 7C3C1CFF 42B17218
	s_wait_alu 0xfffd                                          // 000000004988: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000498C: BF8700A2
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo             // 000000004990: 021C1EFF 7F800000
	s_wait_loadcnt 0x0                                         // 000000004998: BFC00000
	v_dual_fmac_f32 v5, v14, v13 :: v_dual_add_f32 v4, v4, v14 // 00000000499C: C8081B0E 05041D04
	s_wait_alu 0xfffe                                          // 0000000049A4: BF88FFFE
	s_and_not1_saveexec_b32 s36, s36                           // 0000000049A8: BEA43024
	s_cbranch_execz 65250                                      // 0000000049AC: BFA5FEE2 <ullm_paged_decode_attn_f32_kernel+0x1b38>
	v_sub_f32_e32 v12, v12, v14                                // 0000000049B0: 08181D0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000049B4: BF870091
	v_mul_f32_e32 v15, 0x3fb8aa3b, v12                         // 0000000049B8: 101E18FF 3FB8AA3B
	v_fma_f32 v16, 0x3fb8aa3b, v12, -v15                       // 0000000049C0: D6130010 843E18FF 3FB8AA3B
	v_rndne_f32_e32 v17, v15                                   // 0000000049CC: 7E22470F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000049D0: BF870221
	v_sub_f32_e32 v15, v15, v17                                // 0000000049D4: 081E230F
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v12                  // 0000000049D8: 7C3618FF C2CE8ED0
	v_fmac_f32_e32 v16, 0x32a5705f, v12                        // 0000000049E0: 562018FF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000049E8: BF870121
	v_add_f32_e32 v15, v15, v16                                // 0000000049EC: 061E210F
	v_cvt_i32_f32_e32 v16, v17                                 // 0000000049F0: 7E201111
	v_exp_f32_e32 v15, v15                                     // 0000000049F4: 7E1E4B0F
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000049F8: BF8700A5
	v_ldexp_f32 v15, v15, v16                                  // 0000000049FC: D71C000F 0002210F
	s_wait_alu 0xfffd                                          // 000000004A04: BF88FFFD
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo                      // 000000004A08: 021E1E80
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v12                  // 000000004A0C: 7C3C18FF 42B17218
	s_wait_alu 0xfffd                                          // 000000004A14: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004A18: BF8700A2
	v_cndmask_b32_e32 v12, 0x7f800000, v15, vcc_lo             // 000000004A1C: 02181EFF 7F800000
	s_wait_loadcnt 0x0                                         // 000000004A24: BFC00000
	v_fmac_f32_e32 v13, v5, v12                                // 000000004A28: 561A1905
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004A2C: BF870001
	v_mov_b32_e32 v5, v13                                      // 000000004A30: 7E0A030D
	v_fma_f32 v4, v4, v12, 1.0                                 // 000000004A34: D6130004 03CA1904
	v_mov_b32_e32 v12, v14                                     // 000000004A3C: 7E18030E
	s_branch 65213                                             // 000000004A40: BFA0FEBD <ullm_paged_decode_attn_f32_kernel+0x1b38>
	s_cbranch_execz 65226                                      // 000000004A44: BFA5FECA <ullm_paged_decode_attn_f32_kernel+0x1b70>
	s_and_b32 vcc_lo, exec_lo, s36                             // 000000004A48: 8B6A247E
	s_mov_b32 s0, -1                                           // 000000004A4C: BE8000C1
	s_wait_alu 0xfffe                                          // 000000004A50: BF88FFFE
	s_cbranch_vccz 135                                         // 000000004A54: BFA30087 <ullm_paged_decode_attn_f32_kernel+0x2274>
	s_and_saveexec_b32 s0, s1                                  // 000000004A58: BE802001
	s_cbranch_execz 10                                         // 000000004A5C: BFA5000A <ullm_paged_decode_attn_f32_kernel+0x2088>
	s_mul_u64 s[2:3], s[42:43], s[10:11]                       // 000000004A60: AA820A2A
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0       // 000000004A64: CA220080 03020082
	s_wait_alu 0xfffe                                          // 000000004A6C: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000004A70: 84828202
	s_wait_alu 0xfffe                                          // 000000004A74: BF88FFFE
	s_add_nc_u64 s[2:3], s[28:29], s[2:3]                      // 000000004A78: A982021C
	global_store_b32 v2, v3, s[2:3]                            // 000000004A7C: EE068002 01800000 00000002
	s_wait_alu 0xfffe                                          // 000000004A88: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004A8C: 8C7E007E
	s_mov_b32 s0, 0                                            // 000000004A90: BE800080
	s_branch 6                                                 // 000000004A94: BFA00006 <ullm_paged_decode_attn_f32_kernel+0x20b0>
	s_branch 65023                                             // 000000004A98: BFA0FDFF <ullm_paged_decode_attn_f32_kernel+0x1898>
	s_mov_b32 s0, 0                                            // 000000004A9C: BE800080
	s_cbranch_execz 3                                          // 000000004AA0: BFA50003 <ullm_paged_decode_attn_f32_kernel+0x20b0>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0               // 000000004AA4: CA100080 04040080
	s_mov_b32 s0, -1                                           // 000000004AAC: BE8000C1
	s_wait_alu 0xfffe                                          // 000000004AB0: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000004AB4: 916A007E
	s_wait_alu 0xfffe                                          // 000000004AB8: BF88FFFE
	s_cbranch_vccnz 63463                                      // 000000004ABC: BFA4F7E7 <ullm_paged_decode_attn_f32_kernel+0x5c>
	s_mov_b32 s0, exec_lo                                      // 000000004AC0: BE80007E
	v_cmpx_gt_u64_e64 s[42:43], v[0:1]                         // 000000004AC4: D4DC007E 0002002A
	s_cbranch_execz 63459                                      // 000000004ACC: BFA5F7E3 <ullm_paged_decode_attn_f32_kernel+0x5c>
	v_div_scale_f32 v3, null, v4, v4, v5                       // 000000004AD0: D6FC7C03 04160904
	v_div_scale_f32 v7, vcc_lo, v5, v4, v5                     // 000000004AD8: D6FC6A07 04160905
	v_mad_co_u64_u32 v[0:1], null, s42, s10, v[0:1]            // 000000004AE0: D6FE7C00 0400142A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)// 000000004AE8: BF8702A3
	v_rcp_f32_e32 v6, v3                                       // 000000004AEC: 7E0C5503
	s_cmp_eq_u64 s[14:15], 0                                   // 000000004AF0: BF10800E
	v_fma_f32 v2, -v3, v6, 1.0                                 // 000000004AF4: D6130002 23CA0D03
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000004AFC: BF870191
	v_fmac_f32_e32 v6, v2, v6                                  // 000000004B00: 560C0D02
	v_mad_co_u64_u32 v[1:2], null, s43, s10, v[1:2]            // 000000004B04: D6FE7C01 0404142B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004B0C: BF870112
	v_mul_f32_e32 v8, v7, v6                                   // 000000004B10: 10100D07
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000004B14: 3E000082
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004B18: BF870092
	v_fma_f32 v9, -v3, v8, v7                                  // 000000004B1C: D6130009 241E1103
	v_fmac_f32_e32 v8, v9, v6                                  // 000000004B24: 56100D09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004B28: BF8700A1
	v_fma_f32 v2, -v3, v8, v7                                  // 000000004B2C: D6130002 241E1103
	s_wait_alu 0xfffd                                          // 000000004B34: BF88FFFD
	v_div_fmas_f32 v2, v2, v6, v8                              // 000000004B38: D6370002 04220D02
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004B40: BF870001
	v_div_fixup_f32 v2, v2, v4, v5                             // 000000004B44: D6270002 04160902
	s_cbranch_scc1 75                                          // 000000004B4C: BFA2004B <ullm_paged_decode_attn_f32_kernel+0x227c>
	v_add_co_u32 v3, vcc_lo, s14, v0                           // 000000004B50: D7006A03 0002000E
	s_wait_alu 0xfffd                                          // 000000004B58: BF88FFFD
	v_add_co_ci_u32_e64 v4, null, s15, v1, vcc_lo              // 000000004B5C: D5207C04 01AA020F
	global_load_b32 v3, v[3:4], off                            // 000000004B64: EE05007C 00000003 00000003
	s_wait_loadcnt 0x0                                         // 000000004B70: BFC00000
	v_mul_f32_e32 v4, 0xbfb8aa3b, v3                           // 000000004B74: 100806FF BFB8AA3B
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3                   // 000000004B7C: 7C3C06FF 42CE8ED0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004B84: BF8700A2
	v_fma_f32 v5, 0xbfb8aa3b, v3, -v4                          // 000000004B88: D6130005 841206FF BFB8AA3B
	v_rndne_f32_e32 v6, v4                                     // 000000004B94: 7E0C4704
	v_dual_fmamk_f32 v5, v3, 0xb2a5705f, v5 :: v_dual_sub_f32 v4, v4, v6// 000000004B98: C88A0B03 05040D04 B2A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004BA4: BF870121
	v_add_f32_e32 v4, v4, v5                                   // 000000004BA8: 06080B04
	v_cvt_i32_f32_e32 v5, v6                                   // 000000004BAC: 7E0A1106
	v_exp_f32_e32 v4, v4                                       // 000000004BB0: 7E084B04
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004BB4: BF8700A5
	v_ldexp_f32 v4, v4, v5                                     // 000000004BB8: D71C0004 00020B04
	s_wait_alu 0xfffd                                          // 000000004BC0: BF88FFFD
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000004BC4: 02080880
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3                   // 000000004BC8: 7C3606FF C2B17218
	s_wait_alu 0xfffd                                          // 000000004BD0: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004BD4: BF870092
	v_cndmask_b32_e32 v3, 0x7f800000, v4, vcc_lo               // 000000004BD8: 020608FF 7F800000
	v_add_f32_e32 v3, 1.0, v3                                  // 000000004BE0: 060606F2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004BE4: BF870121
	v_div_scale_f32 v4, null, v3, v3, 1.0                      // 000000004BE8: D6FC7C04 03CA0703
	v_div_scale_f32 v7, vcc_lo, 1.0, v3, 1.0                   // 000000004BF0: D6FC6A07 03CA06F2
	v_rcp_f32_e32 v5, v4                                       // 000000004BF8: 7E0A5504
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004BFC: BF870095
	v_fma_f32 v6, -v4, v5, 1.0                                 // 000000004C00: D6130006 23CA0B04
	v_fmac_f32_e32 v5, v6, v5                                  // 000000004C08: 560A0B06
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C0C: BF870091
	v_mul_f32_e32 v6, v7, v5                                   // 000000004C10: 100C0B07
	v_fma_f32 v8, -v4, v6, v7                                  // 000000004C14: D6130008 241E0D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C1C: BF870091
	v_fmac_f32_e32 v6, v8, v5                                  // 000000004C20: 560C0B08
	v_fma_f32 v4, -v4, v6, v7                                  // 000000004C24: D6130004 241E0D04
	s_wait_alu 0xfffd                                          // 000000004C2C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C30: BF870091
	v_div_fmas_f32 v4, v4, v5, v6                              // 000000004C34: D6370004 041A0B04
	v_div_fixup_f32 v3, v4, v3, 1.0                            // 000000004C3C: D6270003 03CA0704
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004C44: BF870001
	v_mul_f32_e32 v5, v2, v3                                   // 000000004C48: 100A0702
	v_add_co_u32 v3, vcc_lo, s28, v0                           // 000000004C4C: D7006A03 0002001C
	s_wait_alu 0xfffd                                          // 000000004C54: BF88FFFD
	v_add_co_ci_u32_e64 v4, null, s29, v1, vcc_lo              // 000000004C58: D5207C04 01AA021D
	global_store_b32 v[3:4], v5, off                           // 000000004C60: EE06807C 02800000 00000003
	s_cbranch_execnz 63355                                     // 000000004C6C: BFA6F77B <ullm_paged_decode_attn_f32_kernel+0x5c>
	s_branch 2                                                 // 000000004C70: BFA00002 <ullm_paged_decode_attn_f32_kernel+0x227c>
	s_branch 65422                                             // 000000004C74: BFA0FF8E <ullm_paged_decode_attn_f32_kernel+0x20b0>
	s_branch 64751                                             // 000000004C78: BFA0FCEF <ullm_paged_decode_attn_f32_kernel+0x1638>
	v_add_co_u32 v0, vcc_lo, s28, v0                           // 000000004C7C: D7006A00 0002001C
	s_wait_alu 0xfffd                                          // 000000004C84: BF88FFFD
	v_add_co_ci_u32_e64 v1, null, s29, v1, vcc_lo              // 000000004C88: D5207C01 01AA021D
	global_store_b32 v[0:1], v2, off                           // 000000004C90: EE06807C 01000000 00000000
	s_endpgm                                                   // 000000004C9C: BFB00000
	s_nop 0                                                    // 000000004CA0: BF800000
	s_nop 0                                                    // 000000004CA4: BF800000
	s_nop 0                                                    // 000000004CA8: BF800000
	s_nop 0                                                    // 000000004CAC: BF800000
	s_nop 0                                                    // 000000004CB0: BF800000
	s_nop 0                                                    // 000000004CB4: BF800000
	s_nop 0                                                    // 000000004CB8: BF800000
	s_nop 0                                                    // 000000004CBC: BF800000
	s_nop 0                                                    // 000000004CC0: BF800000
	s_nop 0                                                    // 000000004CC4: BF800000
	s_nop 0                                                    // 000000004CC8: BF800000
	s_nop 0                                                    // 000000004CCC: BF800000
	s_nop 0                                                    // 000000004CD0: BF800000
	s_nop 0                                                    // 000000004CD4: BF800000
	s_nop 0                                                    // 000000004CD8: BF800000
	s_nop 0                                                    // 000000004CDC: BF800000
	s_nop 0                                                    // 000000004CE0: BF800000
	s_nop 0                                                    // 000000004CE4: BF800000
	s_nop 0                                                    // 000000004CE8: BF800000
	s_nop 0                                                    // 000000004CEC: BF800000
	s_nop 0                                                    // 000000004CF0: BF800000
	s_nop 0                                                    // 000000004CF4: BF800000
	s_nop 0                                                    // 000000004CF8: BF800000
	s_nop 0                                                    // 000000004CFC: BF800000
