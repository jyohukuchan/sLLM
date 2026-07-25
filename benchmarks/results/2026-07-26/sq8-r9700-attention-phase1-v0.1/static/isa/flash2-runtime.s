
benchmarks/results/2026-07-26/sq8-r9700-attention-phase1-v0.1/prefill/code-objects/runtime-dump/_code_object0006.o:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001b00 <ullm_cached_prefix_attn_f32_flash2_kernel>:
	s_load_b512 s[8:23], s[0:1], 0x0                           // 000000001B00: F4008200 F8000000
	s_mov_b32 s27, 0                                           // 000000001B08: BE9B0080
	s_mov_b32 s24, ttmp9                                       // 000000001B0C: BE980075
	s_mov_b32 s25, s27                                         // 000000001B10: BE99001B
	s_wait_kmcnt 0x0                                           // 000000001B14: BFC70000
	s_mul_u64 s[2:3], s[18:19], s[16:17]                       // 000000001B18: AA821012
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B1C: BF870009
	v_cmp_le_u64_e64 s2, s[2:3], s[24:25]                      // 000000001B20: D45B0002 00003002
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001B28: 8B6A027E
	s_cbranch_vccnz 1020                                       // 000000001B2C: BFA403FC <ullm_cached_prefix_attn_f32_flash2_kernel+0x1020>
	s_clause 0x1                                               // 000000001B30: BF850001
	s_load_b32 s2, s[0:1], 0x64                                // 000000001B34: F4000080 F8000064
	s_load_b64 s[16:17], s[0:1], 0x40                          // 000000001B3C: F4002400 F8000040
	s_wait_kmcnt 0x0                                           // 000000001B44: BFC70000
	s_and_b32 s26, s2, 0xffff                                  // 000000001B48: 8B1AFF02 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B50: BF870009
	v_cmp_gt_u64_e64 s2, s[16:17], s[26:27]                    // 000000001B54: D45C0002 00003410
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001B5C: 8B6A027E
	s_cbranch_vccnz 1007                                       // 000000001B60: BFA403EF <ullm_cached_prefix_attn_f32_flash2_kernel+0x1020>
	v_cmp_lt_u64_e64 s2, s[24:25], s[18:19]                    // 000000001B64: D4590002 00002418
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001B6C: 8B6A027E
	s_mov_b64 s[2:3], 0                                        // 000000001B70: BE820180
	s_cbranch_vccnz 32                                         // 000000001B74: BFA40020 <ullm_cached_prefix_attn_f32_flash2_kernel+0xf8>
	v_cvt_f32_u32_e32 v1, s18                                  // 000000001B78: 7E020C12
	s_sub_co_i32 s3, 0, s18                                    // 000000001B7C: 81831280
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000001B80: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001B84: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001B88: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B90: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001B94: 7E020F01
	v_readfirstlane_b32 s2, v1                                 // 000000001B98: 7E040501
	s_wait_alu 0xfffe                                          // 000000001B9C: BF88FFFE
	s_mul_i32 s3, s3, s2                                       // 000000001BA0: 96030203
	s_wait_alu 0xfffe                                          // 000000001BA4: BF88FFFE
	s_mul_hi_u32 s3, s2, s3                                    // 000000001BA8: 96830302
	s_wait_alu 0xfffe                                          // 000000001BAC: BF88FFFE
	s_add_co_i32 s2, s2, s3                                    // 000000001BB0: 81020302
	s_wait_alu 0xfffe                                          // 000000001BB4: BF88FFFE
	s_mul_hi_u32 s2, s24, s2                                   // 000000001BB8: 96820218
	s_wait_alu 0xfffe                                          // 000000001BBC: BF88FFFE
	s_mul_i32 s3, s2, s18                                      // 000000001BC0: 96031202
	s_add_co_i32 s4, s2, 1                                     // 000000001BC4: 81048102
	s_wait_alu 0xfffe                                          // 000000001BC8: BF88FFFE
	s_sub_co_i32 s3, s24, s3                                   // 000000001BCC: 81830318
	s_wait_alu 0xfffe                                          // 000000001BD0: BF88FFFE
	s_sub_co_i32 s5, s3, s18                                   // 000000001BD4: 81851203
	s_cmp_ge_u32 s3, s18                                       // 000000001BD8: BF091203
	s_cselect_b32 s2, s4, s2                                   // 000000001BDC: 98020204
	s_cselect_b32 s3, s5, s3                                   // 000000001BE0: 98030305
	s_wait_alu 0xfffe                                          // 000000001BE4: BF88FFFE
	s_add_co_i32 s4, s2, 1                                     // 000000001BE8: 81048102
	s_cmp_ge_u32 s3, s18                                       // 000000001BEC: BF091203
	s_mov_b32 s3, 0                                            // 000000001BF0: BE830080
	s_cselect_b32 s2, s4, s2                                   // 000000001BF4: 98020204
	s_or_b64 s[6:7], s[18:19], s[20:21]                        // 000000001BF8: 8C861412
	s_mov_b32 s6, 0                                            // 000000001BFC: BE860080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C00: BF870009
	s_cmp_lg_u64 s[6:7], 0                                     // 000000001C04: BF118006
	s_cbranch_scc0 966                                         // 000000001C08: BFA103C6 <ullm_cached_prefix_attn_f32_flash2_kernel+0x1024>
	s_cvt_f32_u32 s4, s20                                      // 000000001C0C: BE846514
	s_cvt_f32_u32 s5, s21                                      // 000000001C10: BE856515
	s_sub_nc_u64 s[28:29], 0, s[20:21]                         // 000000001C14: AA1C1480
	s_mov_b32 s31, s6                                          // 000000001C18: BE9F0006
	s_mov_b32 s37, s6                                          // 000000001C1C: BEA50006
	s_fmamk_f32 s4, s5, 0x4f800000, s4                         // 000000001C20: A3040405 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000001C28: BF87029B
	v_s_rcp_f32 s4, s4                                         // 000000001C2C: D6840004 00000004
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000001C34: A204FF04 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000001C3C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000001C40: BF87052A
	s_mul_f32 s5, s4, 0x2f800000                               // 000000001C44: A205FF04 2F800000
	s_wait_alu 0xfffe                                          // 000000001C4C: BF88FFFE
	s_trunc_f32 s5, s5                                         // 000000001C50: BE856205
	s_wait_alu 0xfffe                                          // 000000001C54: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001C58: BF8704BA
	s_fmamk_f32 s4, s5, 0xcf800000, s4                         // 000000001C5C: A3040405 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000001C64: BE856705
	s_wait_alu 0xfffe                                          // 000000001C68: BF88FFFE
	s_cvt_u32_f32 s4, s4                                       // 000000001C6C: BE846704
	s_wait_alu 0xfffe                                          // 000000001C70: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C74: BF87049A
	s_mul_u64 s[34:35], s[28:29], s[4:5]                       // 000000001C78: AAA2041C
	s_mul_hi_u32 s39, s4, s35                                  // 000000001C7C: 96A72304
	s_mul_i32 s38, s4, s35                                     // 000000001C80: 96262304
	s_mul_hi_u32 s30, s4, s34                                  // 000000001C84: 969E2204
	s_mul_i32 s27, s5, s34                                     // 000000001C88: 961B2205
	s_add_nc_u64 s[30:31], s[30:31], s[38:39]                  // 000000001C8C: A99E261E
	s_mul_hi_u32 s7, s5, s34                                   // 000000001C90: 96872205
	s_mul_hi_u32 s33, s5, s35                                  // 000000001C94: 96A12305
	s_wait_alu 0xfffe                                          // 000000001C98: BF88FFFE
	s_add_co_u32 s27, s30, s27                                 // 000000001C9C: 801B1B1E
	s_add_co_ci_u32 s36, s31, s7                               // 000000001CA0: 8224071F
	s_mul_i32 s34, s5, s35                                     // 000000001CA4: 96222305
	s_add_co_ci_u32 s35, s33, 0                                // 000000001CA8: 82238021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001CAC: BF8704C9
	s_add_nc_u64 s[30:31], s[36:37], s[34:35]                  // 000000001CB0: A99E2224
	s_mov_b32 s35, s6                                          // 000000001CB4: BEA30006
	s_add_co_u32 s4, s4, s30                                   // 000000001CB8: 80041E04
	s_cselect_b32 s7, -1, 0                                    // 000000001CBC: 980780C1
	s_cmp_lg_u32 s7, 0                                         // 000000001CC0: BF078007
	s_add_co_ci_u32 s5, s5, s31                                // 000000001CC4: 82051F05
	s_mov_b32 s31, s6                                          // 000000001CC8: BE9F0006
	s_wait_alu 0xfffe                                          // 000000001CCC: BF88FFFE
	s_mul_u64 s[28:29], s[28:29], s[4:5]                       // 000000001CD0: AA9C041C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CD4: BF870009
	s_mul_hi_u32 s37, s4, s29                                  // 000000001CD8: 96A51D04
	s_mul_i32 s36, s4, s29                                     // 000000001CDC: 96241D04
	s_mul_hi_u32 s30, s4, s28                                  // 000000001CE0: 969E1C04
	s_mul_i32 s27, s5, s28                                     // 000000001CE4: 961B1C05
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]                  // 000000001CE8: A99E241E
	s_mul_hi_u32 s7, s5, s28                                   // 000000001CEC: 96871C05
	s_mul_hi_u32 s33, s5, s29                                  // 000000001CF0: 96A11D05
	s_wait_alu 0xfffe                                          // 000000001CF4: BF88FFFE
	s_add_co_u32 s27, s30, s27                                 // 000000001CF8: 801B1B1E
	s_add_co_ci_u32 s34, s31, s7                               // 000000001CFC: 8222071F
	s_mul_i32 s28, s5, s29                                     // 000000001D00: 961C1D05
	s_add_co_ci_u32 s29, s33, 0                                // 000000001D04: 821D8021
	s_mov_b32 s31, s6                                          // 000000001D08: BE9F0006
	s_add_nc_u64 s[28:29], s[34:35], s[28:29]                  // 000000001D0C: A99C1C22
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D10: BF870009
	s_add_co_u32 s4, s4, s28                                   // 000000001D14: 80041C04
	s_cselect_b32 s7, -1, 0                                    // 000000001D18: 980780C1
	s_wait_alu 0xfffe                                          // 000000001D1C: BF88FFFE
	s_mul_hi_u32 s30, s18, s4                                  // 000000001D20: 969E0412
	s_cmp_lg_u32 s7, 0                                         // 000000001D24: BF078007
	s_mul_hi_u32 s7, s19, s4                                   // 000000001D28: 96870413
	s_add_co_ci_u32 s27, s5, s29                               // 000000001D2C: 821B1D05
	s_mul_i32 s29, s19, s4                                     // 000000001D30: 961D0413
	s_wait_alu 0xfffe                                          // 000000001D34: BF88FFFE
	s_mul_hi_u32 s5, s18, s27                                  // 000000001D38: 96851B12
	s_mul_i32 s4, s18, s27                                     // 000000001D3C: 96041B12
	s_mul_hi_u32 s33, s19, s27                                 // 000000001D40: 96A11B13
	s_wait_alu 0xfffe                                          // 000000001D44: BF88FFFE
	s_add_nc_u64 s[4:5], s[30:31], s[4:5]                      // 000000001D48: A984041E
	s_mul_i32 s28, s19, s27                                    // 000000001D4C: 961C1B13
	s_wait_alu 0xfffe                                          // 000000001D50: BF88FFFE
	s_add_co_u32 s4, s4, s29                                   // 000000001D54: 80041D04
	s_add_co_ci_u32 s34, s5, s7                                // 000000001D58: 82220705
	s_add_co_ci_u32 s29, s33, 0                                // 000000001D5C: 821D8021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001D60: BF8704B9
	s_add_nc_u64 s[4:5], s[34:35], s[28:29]                    // 000000001D64: A9841C22
	s_wait_alu 0xfffe                                          // 000000001D68: BF88FFFE
	s_mul_u64 s[28:29], s[20:21], s[4:5]                       // 000000001D6C: AA9C0414
	s_sub_co_u32 s7, s18, s28                                  // 000000001D70: 80871C12
	s_cselect_b32 s27, -1, 0                                   // 000000001D74: 981B80C1
	s_sub_co_i32 s28, s19, s29                                 // 000000001D78: 819C1D13
	s_wait_alu 0xfffe                                          // 000000001D7C: BF88FFFE
	s_cmp_lg_u32 s27, 0                                        // 000000001D80: BF07801B
	s_sub_co_ci_u32 s28, s28, s21                              // 000000001D84: 829C151C
	s_sub_co_u32 s30, s7, s20                                  // 000000001D88: 809E1407
	s_cselect_b32 s31, -1, 0                                   // 000000001D8C: 981F80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001D90: BF8704A9
	s_cmp_lg_u32 s31, 0                                        // 000000001D94: BF07801F
	s_sub_co_ci_u32 s28, s28, 0                                // 000000001D98: 829C801C
	s_cmp_ge_u32 s28, s21                                      // 000000001D9C: BF09151C
	s_cselect_b32 s33, -1, 0                                   // 000000001DA0: 982180C1
	s_cmp_ge_u32 s30, s20                                      // 000000001DA4: BF09141E
	s_add_nc_u64 s[30:31], s[4:5], 1                           // 000000001DA8: A99E8104
	s_cselect_b32 s34, -1, 0                                   // 000000001DAC: 982280C1
	s_cmp_eq_u32 s28, s21                                      // 000000001DB0: BF06151C
	s_cselect_b32 s28, s34, s33                                // 000000001DB4: 981C2122
	s_add_nc_u64 s[34:35], s[4:5], 2                           // 000000001DB8: A9A28204
	s_cmp_lg_u32 s28, 0                                        // 000000001DBC: BF07801C
	s_cselect_b32 s28, s34, s30                                // 000000001DC0: 981C1E22
	s_cselect_b32 s30, s35, s31                                // 000000001DC4: 981E1F23
	s_cmp_lg_u32 s27, 0                                        // 000000001DC8: BF07801B
	s_sub_co_ci_u32 s27, s19, s29                              // 000000001DCC: 829B1D13
	s_wait_alu 0xfffe                                          // 000000001DD0: BF88FFFE
	s_cmp_ge_u32 s27, s21                                      // 000000001DD4: BF09151B
	s_cselect_b32 s29, -1, 0                                   // 000000001DD8: 981D80C1
	s_cmp_ge_u32 s7, s20                                       // 000000001DDC: BF091407
	s_cselect_b32 s7, -1, 0                                    // 000000001DE0: 980780C1
	s_cmp_eq_u32 s27, s21                                      // 000000001DE4: BF06151B
	s_cselect_b32 s7, s7, s29                                  // 000000001DE8: 98071D07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DEC: BF870009
	s_cmp_lg_u32 s7, 0                                         // 000000001DF0: BF078007
	s_cselect_b32 s5, s30, s5                                  // 000000001DF4: 9805051E
	s_cselect_b32 s4, s28, s4                                  // 000000001DF8: 9804041C
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000001DFC: 916A067E
	s_cbranch_vccnz 32                                         // 000000001E00: BFA40020 <ullm_cached_prefix_attn_f32_flash2_kernel+0x384>
	v_cvt_f32_u32_e32 v1, s20                                  // 000000001E04: 7E020C14
	s_sub_co_i32 s5, 0, s20                                    // 000000001E08: 81851480
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000001E0C: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001E10: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001E14: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E1C: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001E20: 7E020F01
	v_readfirstlane_b32 s4, v1                                 // 000000001E24: 7E080501
	s_wait_alu 0xfffe                                          // 000000001E28: BF88FFFE
	s_mul_i32 s5, s5, s4                                       // 000000001E2C: 96050405
	s_wait_alu 0xfffe                                          // 000000001E30: BF88FFFE
	s_mul_hi_u32 s5, s4, s5                                    // 000000001E34: 96850504
	s_wait_alu 0xfffe                                          // 000000001E38: BF88FFFE
	s_add_co_i32 s4, s4, s5                                    // 000000001E3C: 81040504
	s_wait_alu 0xfffe                                          // 000000001E40: BF88FFFE
	s_mul_hi_u32 s4, s18, s4                                   // 000000001E44: 96840412
	s_wait_alu 0xfffe                                          // 000000001E48: BF88FFFE
	s_mul_i32 s5, s4, s20                                      // 000000001E4C: 96051404
	s_add_co_i32 s6, s4, 1                                     // 000000001E50: 81068104
	s_wait_alu 0xfffe                                          // 000000001E54: BF88FFFE
	s_sub_co_i32 s5, s18, s5                                   // 000000001E58: 81850512
	s_wait_alu 0xfffe                                          // 000000001E5C: BF88FFFE
	s_sub_co_i32 s7, s5, s20                                   // 000000001E60: 81871405
	s_cmp_ge_u32 s5, s20                                       // 000000001E64: BF091405
	s_cselect_b32 s4, s6, s4                                   // 000000001E68: 98040406
	s_cselect_b32 s5, s7, s5                                   // 000000001E6C: 98050507
	s_wait_alu 0xfffe                                          // 000000001E70: BF88FFFE
	s_add_co_i32 s6, s4, 1                                     // 000000001E74: 81068104
	s_cmp_ge_u32 s5, s20                                       // 000000001E78: BF091405
	s_mov_b32 s5, 0                                            // 000000001E7C: BE850080
	s_cselect_b32 s4, s6, s4                                   // 000000001E80: 98040406
	s_mul_u64 s[6:7], s[2:3], s[18:19]                         // 000000001E84: AA861202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001E88: BF8704C9
	s_sub_nc_u64 s[6:7], s[24:25], s[6:7]                      // 000000001E8C: AA060618
	s_wait_alu 0xfffe                                          // 000000001E90: BF88FFFE
	s_or_b64 s[28:29], s[6:7], s[4:5]                          // 000000001E94: 8C9C0406
	s_mov_b32 s28, 0                                           // 000000001E98: BE9C0080
	s_cmp_lg_u64 s[28:29], 0                                   // 000000001E9C: BF11801C
	s_cbranch_scc0 801                                         // 000000001EA0: BFA10321 <ullm_cached_prefix_attn_f32_flash2_kernel+0x1028>
	s_cvt_f32_u32 s18, s4                                      // 000000001EA4: BE926504
	s_cvt_f32_u32 s19, s5                                      // 000000001EA8: BE936505
	s_sub_nc_u64 s[30:31], 0, s[4:5]                           // 000000001EAC: AA1E0480
	s_mov_b32 s35, s28                                         // 000000001EB0: BEA3001C
	s_mov_b32 s39, s28                                         // 000000001EB4: BEA7001C
	s_wait_alu 0xfffe                                          // 000000001EB8: BF88FFFE
	s_fmamk_f32 s18, s19, 0x4f800000, s18                      // 000000001EBC: A3121213 4F800000
	s_wait_alu 0xfffe                                          // 000000001EC4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000001EC8: BF87029A
	v_s_rcp_f32 s18, s18                                       // 000000001ECC: D6840012 00000012
	s_mul_f32 s18, s18, 0x5f7ffffc                             // 000000001ED4: A212FF12 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000001EDC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000001EE0: BF87052A
	s_mul_f32 s19, s18, 0x2f800000                             // 000000001EE4: A213FF12 2F800000
	s_wait_alu 0xfffe                                          // 000000001EEC: BF88FFFE
	s_trunc_f32 s19, s19                                       // 000000001EF0: BE936213
	s_wait_alu 0xfffe                                          // 000000001EF4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001EF8: BF8704BA
	s_fmamk_f32 s18, s19, 0xcf800000, s18                      // 000000001EFC: A3121213 CF800000
	s_cvt_u32_f32 s19, s19                                     // 000000001F04: BE936713
	s_wait_alu 0xfffe                                          // 000000001F08: BF88FFFE
	s_cvt_u32_f32 s18, s18                                     // 000000001F0C: BE926712
	s_wait_alu 0xfffe                                          // 000000001F10: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F14: BF87049A
	s_mul_u64 s[36:37], s[30:31], s[18:19]                     // 000000001F18: AAA4121E
	s_mul_hi_u32 s41, s18, s37                                 // 000000001F1C: 96A92512
	s_mul_i32 s40, s18, s37                                    // 000000001F20: 96282512
	s_mul_hi_u32 s34, s18, s36                                 // 000000001F24: 96A22412
	s_mul_i32 s29, s19, s36                                    // 000000001F28: 961D2413
	s_add_nc_u64 s[34:35], s[34:35], s[40:41]                  // 000000001F2C: A9A22822
	s_mul_hi_u32 s27, s19, s36                                 // 000000001F30: 969B2413
	s_mul_hi_u32 s33, s19, s37                                 // 000000001F34: 96A12513
	s_add_co_u32 s29, s34, s29                                 // 000000001F38: 801D1D22
	s_wait_alu 0xfffe                                          // 000000001F3C: BF88FFFE
	s_add_co_ci_u32 s38, s35, s27                              // 000000001F40: 82261B23
	s_mul_i32 s36, s19, s37                                    // 000000001F44: 96242513
	s_add_co_ci_u32 s37, s33, 0                                // 000000001F48: 82258021
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F4C: BF870009
	s_add_nc_u64 s[34:35], s[38:39], s[36:37]                  // 000000001F50: A9A22426
	s_mov_b32 s37, s28                                         // 000000001F54: BEA5001C
	s_add_co_u32 s18, s18, s34                                 // 000000001F58: 80122212
	s_cselect_b32 s27, -1, 0                                   // 000000001F5C: 981B80C1
	s_wait_alu 0xfffe                                          // 000000001F60: BF88FFFE
	s_cmp_lg_u32 s27, 0                                        // 000000001F64: BF07801B
	s_add_co_ci_u32 s19, s19, s35                              // 000000001F68: 82132313
	s_mov_b32 s35, s28                                         // 000000001F6C: BEA3001C
	s_wait_alu 0xfffe                                          // 000000001F70: BF88FFFE
	s_mul_u64 s[30:31], s[30:31], s[18:19]                     // 000000001F74: AA9E121E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F78: BF870009
	s_mul_hi_u32 s39, s18, s31                                 // 000000001F7C: 96A71F12
	s_mul_i32 s38, s18, s31                                    // 000000001F80: 96261F12
	s_mul_hi_u32 s34, s18, s30                                 // 000000001F84: 96A21E12
	s_mul_i32 s29, s19, s30                                    // 000000001F88: 961D1E13
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]                  // 000000001F8C: A9A22622
	s_mul_hi_u32 s27, s19, s30                                 // 000000001F90: 969B1E13
	s_mul_hi_u32 s33, s19, s31                                 // 000000001F94: 96A11F13
	s_add_co_u32 s29, s34, s29                                 // 000000001F98: 801D1D22
	s_wait_alu 0xfffe                                          // 000000001F9C: BF88FFFE
	s_add_co_ci_u32 s36, s35, s27                              // 000000001FA0: 82241B23
	s_mul_i32 s30, s19, s31                                    // 000000001FA4: 961E1F13
	s_add_co_ci_u32 s31, s33, 0                                // 000000001FA8: 821F8021
	s_mov_b32 s35, s28                                         // 000000001FAC: BEA3001C
	s_add_nc_u64 s[30:31], s[36:37], s[30:31]                  // 000000001FB0: A99E1E24
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001FB4: BF870009
	s_add_co_u32 s18, s18, s30                                 // 000000001FB8: 80121E12
	s_cselect_b32 s27, -1, 0                                   // 000000001FBC: 981B80C1
	s_wait_alu 0xfffe                                          // 000000001FC0: BF88FFFE
	s_mul_hi_u32 s34, s6, s18                                  // 000000001FC4: 96A21206
	s_cmp_lg_u32 s27, 0                                        // 000000001FC8: BF07801B
	s_mul_hi_u32 s27, s7, s18                                  // 000000001FCC: 969B1207
	s_add_co_ci_u32 s29, s19, s31                              // 000000001FD0: 821D1F13
	s_mul_i32 s31, s7, s18                                     // 000000001FD4: 961F1207
	s_mul_hi_u32 s19, s6, s29                                  // 000000001FD8: 96931D06
	s_mul_i32 s18, s6, s29                                     // 000000001FDC: 96121D06
	s_mul_hi_u32 s33, s7, s29                                  // 000000001FE0: 96A11D07
	s_wait_alu 0xfffe                                          // 000000001FE4: BF88FFFE
	s_add_nc_u64 s[18:19], s[34:35], s[18:19]                  // 000000001FE8: A9921222
	s_mul_i32 s30, s7, s29                                     // 000000001FEC: 961E1D07
	s_wait_alu 0xfffe                                          // 000000001FF0: BF88FFFE
	s_add_co_u32 s18, s18, s31                                 // 000000001FF4: 80121F12
	s_add_co_ci_u32 s36, s19, s27                              // 000000001FF8: 82241B13
	s_add_co_ci_u32 s31, s33, 0                                // 000000001FFC: 821F8021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002000: BF8704B9
	s_add_nc_u64 s[18:19], s[36:37], s[30:31]                  // 000000002004: A9921E24
	s_wait_alu 0xfffe                                          // 000000002008: BF88FFFE
	s_mul_u64 s[30:31], s[4:5], s[18:19]                       // 00000000200C: AA9E1204
	s_sub_co_u32 s27, s6, s30                                  // 000000002010: 809B1E06
	s_cselect_b32 s29, -1, 0                                   // 000000002014: 981D80C1
	s_sub_co_i32 s30, s7, s31                                  // 000000002018: 819E1F07
	s_cmp_lg_u32 s29, 0                                        // 00000000201C: BF07801D
	s_sub_co_ci_u32 s30, s30, s5                               // 000000002020: 829E051E
	s_wait_alu 0xfffe                                          // 000000002024: BF88FFFE
	s_sub_co_u32 s33, s27, s4                                  // 000000002028: 80A1041B
	s_cselect_b32 s34, -1, 0                                   // 00000000202C: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000002030: BF8704B9
	s_cmp_lg_u32 s34, 0                                        // 000000002034: BF078022
	s_add_nc_u64 s[34:35], s[18:19], 1                         // 000000002038: A9A28112
	s_sub_co_ci_u32 s30, s30, 0                                // 00000000203C: 829E801E
	s_cmp_ge_u32 s30, s5                                       // 000000002040: BF09051E
	s_cselect_b32 s36, -1, 0                                   // 000000002044: 982480C1
	s_cmp_ge_u32 s33, s4                                       // 000000002048: BF090421
	s_cselect_b32 s33, -1, 0                                   // 00000000204C: 982180C1
	s_cmp_eq_u32 s30, s5                                       // 000000002050: BF06051E
	s_cselect_b32 s30, s33, s36                                // 000000002054: 981E2421
	s_add_nc_u64 s[36:37], s[18:19], 2                         // 000000002058: A9A48212
	s_cmp_lg_u32 s30, 0                                        // 00000000205C: BF07801E
	s_cselect_b32 s30, s36, s34                                // 000000002060: 981E2224
	s_cselect_b32 s33, s37, s35                                // 000000002064: 98212325
	s_cmp_lg_u32 s29, 0                                        // 000000002068: BF07801D
	s_sub_co_ci_u32 s7, s7, s31                                // 00000000206C: 82871F07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002070: BF870009
	s_cmp_ge_u32 s7, s5                                        // 000000002074: BF090507
	s_cselect_b32 s29, -1, 0                                   // 000000002078: 981D80C1
	s_cmp_ge_u32 s27, s4                                       // 00000000207C: BF09041B
	s_cselect_b32 s27, -1, 0                                   // 000000002080: 981B80C1
	s_cmp_eq_u32 s7, s5                                        // 000000002084: BF060507
	s_wait_alu 0xfffe                                          // 000000002088: BF88FFFE
	s_cselect_b32 s5, s27, s29                                 // 00000000208C: 98051D1B
	s_wait_alu 0xfffe                                          // 000000002090: BF88FFFE
	s_cmp_lg_u32 s5, 0                                         // 000000002094: BF078005
	s_cselect_b32 s19, s33, s19                                // 000000002098: 98131321
	s_cselect_b32 s18, s30, s18                                // 00000000209C: 9812121E
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 0000000020A0: 916A1C7E
	s_cbranch_vccnz 29                                         // 0000000020A4: BFA4001D <ullm_cached_prefix_attn_f32_flash2_kernel+0x61c>
	v_cvt_f32_u32_e32 v1, s4                                   // 0000000020A8: 7E020C04
	s_sub_co_i32 s7, 0, s4                                     // 0000000020AC: 81870480
	s_mov_b32 s19, 0                                           // 0000000020B0: BE930080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000020B4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 0000000020B8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000020BC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000020C4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000020C8: 7E020F01
	v_readfirstlane_b32 s5, v1                                 // 0000000020CC: 7E0A0501
	s_mul_i32 s7, s7, s5                                       // 0000000020D0: 96070507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000020D4: BF870499
	s_mul_hi_u32 s7, s5, s7                                    // 0000000020D8: 96870705
	s_add_co_i32 s5, s5, s7                                    // 0000000020DC: 81050705
	s_wait_alu 0xfffe                                          // 0000000020E0: BF88FFFE
	s_mul_hi_u32 s5, s6, s5                                    // 0000000020E4: 96850506
	s_wait_alu 0xfffe                                          // 0000000020E8: BF88FFFE
	s_mul_i32 s7, s5, s4                                       // 0000000020EC: 96070405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000020F0: BF870009
	s_sub_co_i32 s6, s6, s7                                    // 0000000020F4: 81860706
	s_add_co_i32 s7, s5, 1                                     // 0000000020F8: 81078105
	s_sub_co_i32 s18, s6, s4                                   // 0000000020FC: 81920406
	s_cmp_ge_u32 s6, s4                                        // 000000002100: BF090406
	s_cselect_b32 s5, s7, s5                                   // 000000002104: 98050507
	s_wait_alu 0xfffe                                          // 000000002108: BF88FFFE
	s_cselect_b32 s6, s18, s6                                  // 00000000210C: 98060612
	s_add_co_i32 s7, s5, 1                                     // 000000002110: 81078105
	s_cmp_ge_u32 s6, s4                                        // 000000002114: BF090406
	s_cselect_b32 s18, s7, s5                                  // 000000002118: 98120507
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v8, 2, v0       // 00000000211C: CA220080 01080082
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]                      // 000000002124: A982020E
	s_mov_b64 s[28:29], 0                                      // 000000002128: BE9C0180
	s_wait_alu 0xfffe                                          // 00000000212C: BF88FFFE
	s_add_nc_u64 s[14:15], s[2:3], 1                           // 000000002130: A98E8102
	v_cmp_gt_u64_e64 s2, s[16:17], v[0:1]                      // 000000002134: D45C0002 00020010
	v_cmp_le_u64_e64 s3, s[16:17], v[0:1]                      // 00000000213C: D45B0003 00020010
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v9, v1             // 000000002144: CA100101 04080101
	s_cmp_eq_u64 s[14:15], 0                                   // 00000000214C: BF10800E
	s_cbranch_scc1 591                                         // 000000002150: BFA2024F <ullm_cached_prefix_attn_f32_flash2_kernel+0xf90>
	s_load_b32 s27, s[0:1], 0x48                               // 000000002154: F40006C0 F8000048
	s_cmp_gt_u32 s26, 1                                        // 00000000215C: BF08811A
	s_mul_u64 s[6:7], s[22:23], s[24:25]                       // 000000002160: AA861816
	s_cselect_b32 s33, -1, 0                                   // 000000002164: 982180C1
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000002168: 84868206
	v_dual_mov_b32 v11, 0 :: v_dual_lshlrev_b32 v10, 2, v0     // 00000000216C: CA220080 0B0A0082
	s_add_nc_u64 s[6:7], s[8:9], s[6:7]                        // 000000002174: A9860608
	v_add_co_u32 v12, s12, s12, v8                             // 000000002178: D7000C0C 0002100C
	v_add_co_u32 v2, s6, s6, v8                                // 000000002180: D7000602 00021006
	v_cmp_gt_u64_e64 s4, s[22:23], v[0:1]                      // 000000002188: D45C0004 00020016
	v_cmp_eq_u32_e64 s5, 0, v0                                 // 000000002190: D44A0005 00020080
	s_wait_alu 0xf1ff                                          // 000000002198: BF88F1FF
	v_add_co_ci_u32_e64 v13, null, s13, 0, s12                 // 00000000219C: D5207C0D 0031000D
	v_add_co_ci_u32_e64 v3, null, s7, 0, s6                    // 0000000021A4: D5207C03 00190007
	v_dual_mov_b32 v15, 0 :: v_dual_add_nc_u32 v14, 0x400, v10 // 0000000021AC: CA200080 0F0E14FF 00000400
	v_mov_b32_e32 v9, 0                                        // 0000000021B8: 7E120280
	s_mov_b32 s31, 0                                           // 0000000021BC: BE9F0080
	s_lshl_b32 s34, s26, 2                                     // 0000000021C0: 8422821A
	s_lshl_b32 s35, s26, 2                                     // 0000000021C4: 8423821A
	s_mov_b32 s36, 0xff7fffff                                  // 0000000021C8: BEA400FF FF7FFFFF
	s_sub_nc_u64 s[8:9], s[14:15], s[28:29]                    // 0000000021D0: AA081C0E
	s_mov_b32 s30, s31                                         // 0000000021D4: BE9E001F
	s_wait_alu 0xfffe                                          // 0000000021D8: BF88FFFE
	v_cmp_lt_u64_e64 s6, s[8:9], 64                            // 0000000021DC: D4590006 00018008
	s_and_b32 s6, s6, exec_lo                                  // 0000000021E4: 8B067E06
	s_cselect_b32 s37, s8, 64                                  // 0000000021E8: 9825C008
	s_branch 12                                                // 0000000021EC: BFA0000C <ullm_cached_prefix_attn_f32_flash2_kernel+0x720>
	s_wait_alu 0xfffe                                          // 0000000021F0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000021F4: 8C7E067E
	s_add_co_i32 s30, s30, 1                                   // 0000000021F8: 811E811E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000021FC: BFC80000
	s_wait_alu 0xfffe                                          // 000000002200: BF88FFFE
	s_cmp_ge_u32 s30, s37                                      // 000000002204: BF09251E
	s_barrier_signal -1                                        // 000000002208: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000220C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002210: EE0AC07C 00040000 00000000
	s_cbranch_scc1 109                                         // 00000000221C: BFA2006D <ullm_cached_prefix_attn_f32_flash2_kernel+0x8d4>
	v_mov_b32_e32 v16, 0                                       // 000000002220: 7E200280
	s_and_saveexec_b32 s7, s4                                  // 000000002224: BE872004
	s_cbranch_execz 50                                         // 000000002228: BFA50032 <ullm_cached_prefix_attn_f32_flash2_kernel+0x7f4>
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 00000000222C: A98C1E1C
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v5, v3             // 000000002230: CA100080 10040103
	s_wait_alu 0xfffe                                          // 000000002238: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 00000000223C: AA8C140C
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v7, v1             // 000000002240: CA100102 04060101
	s_wait_alu 0xfffe                                          // 000000002248: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 00000000224C: A98C120C
	v_mov_b32_e32 v6, v0                                       // 000000002250: 7E0C0300
	s_wait_alu 0xfffe                                          // 000000002254: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[22:23]                     // 000000002258: AA8C160C
	s_mov_b32 s38, 0                                           // 00000000225C: BEA60080
	s_wait_alu 0xfffe                                          // 000000002260: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 000000002264: 848C820C
	s_wait_alu 0xfffe                                          // 000000002268: BF88FFFE
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]                  // 00000000226C: A98C0C0A
	v_lshlrev_b64_e32 v[17:18], 2, v[6:7]                      // 000000002270: 3E220C82
	s_wait_alu 0xfffe                                          // 000000002274: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002278: BF870121
	v_add_co_u32 v17, vcc_lo, s12, v17                         // 00000000227C: D7006A11 0002220C
	s_wait_alu 0xfffd                                          // 000000002284: BF88FFFD
	v_add_co_ci_u32_e64 v18, null, s13, v18, vcc_lo            // 000000002288: D5207C12 01AA240D
	v_add_co_u32 v6, vcc_lo, v6, s26                           // 000000002290: D7006A06 00003506
	global_load_b32 v19, v[4:5], off                           // 000000002298: EE05007C 00000013 00000004
	global_load_b32 v17, v[17:18], off                         // 0000000022A4: EE05007C 00000011 00000011
	s_wait_alu 0xfffd                                          // 0000000022B0: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo                // 0000000022B4: D5207C07 01AA0E80
	v_add_co_u32 v4, s6, v4, s34                               // 0000000022BC: D7000604 00004504
	s_wait_alu 0xf1ff                                          // 0000000022C4: BF88F1FF
	v_add_co_ci_u32_e64 v5, null, 0, v5, s6                    // 0000000022C8: D5207C05 001A0A80
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000022D0: BF870003
	v_cmp_le_u64_e32 vcc_lo, s[22:23], v[6:7]                  // 0000000022D4: 7CB60C16
	s_or_b32 s38, vcc_lo, s38                                  // 0000000022D8: 8C26266A
	s_wait_loadcnt 0x0                                         // 0000000022DC: BFC00000
	v_fmac_f32_e32 v16, v19, v17                               // 0000000022E0: 56202313
	s_wait_alu 0xfffe                                          // 0000000022E4: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s38                       // 0000000022E8: 917E267E
	s_cbranch_execnz 65504                                     // 0000000022EC: BFA6FFE0 <ullm_cached_prefix_attn_f32_flash2_kernel+0x770>
	s_or_b32 exec_lo, exec_lo, s38                             // 0000000022F0: 8C7E267E
	s_wait_alu 0xfffe                                          // 0000000022F4: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000022F8: 8C7E077E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000022FC: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000002300: 916A217E
	s_mov_b32 s6, s26                                          // 000000002304: BE86001A
	ds_store_b32 v10, v16                                      // 000000002308: D8340000 0000100A
	s_wait_dscnt 0x0                                           // 000000002310: BFC60000
	s_barrier_signal -1                                        // 000000002314: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002318: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 00000000231C: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000002328: BF88FFFE
	s_cbranch_vccz 23                                          // 00000000232C: BFA30017 <ullm_cached_prefix_attn_f32_flash2_kernel+0x88c>
	s_and_saveexec_b32 s6, s5                                  // 000000002330: BE862005
	s_cbranch_execz 65454                                      // 000000002334: BFA5FFAE <ullm_cached_prefix_attn_f32_flash2_kernel+0x6f0>
	ds_load_b32 v4, v11                                        // 000000002338: D8D80000 0400000B
	s_lshl_b32 s7, s30, 2                                      // 000000002340: 8407821E
	s_wait_dscnt 0x0                                           // 000000002344: BFC60000
	s_wait_kmcnt 0x0                                           // 000000002348: BFC70000
	s_wait_alu 0xfffe                                          // 00000000234C: BF88FFFE
	v_dual_mov_b32 v5, s7 :: v_dual_mul_f32 v4, s27, v4        // 000000002350: CA060007 0504081B
	ds_store_b32 v5, v4 offset:1024                            // 000000002358: D8340400 00000405
	s_branch 65443                                             // 000000002360: BFA0FFA3 <ullm_cached_prefix_attn_f32_flash2_kernel+0x6f0>
	s_or_b32 exec_lo, exec_lo, s12                             // 000000002364: 8C7E0C7E
	s_cmp_lt_u32 s6, 4                                         // 000000002368: BF0A8406
	s_mov_b32 s6, s7                                           // 00000000236C: BE860007
	s_wait_loadcnt_dscnt 0x0                                   // 000000002370: BFC80000
	s_barrier_signal -1                                        // 000000002374: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002378: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 00000000237C: EE0AC07C 00040000 00000000
	s_cbranch_scc1 65513                                       // 000000002388: BFA2FFE9 <ullm_cached_prefix_attn_f32_flash2_kernel+0x830>
	s_wait_alu 0xfffe                                          // 00000000238C: BF88FFFE
	s_lshr_b32 s7, s6, 1                                       // 000000002390: 85078106
	s_mov_b32 s12, exec_lo                                     // 000000002394: BE8C007E
	s_wait_alu 0xfffe                                          // 000000002398: BF88FFFE
	v_cmpx_gt_u32_e64 s7, v0                                   // 00000000239C: D4CC007E 00020007
	s_cbranch_execz 65519                                      // 0000000023A4: BFA5FFEF <ullm_cached_prefix_attn_f32_flash2_kernel+0x864>
	v_lshl_add_u32 v4, s7, 2, v10                              // 0000000023A8: D6460004 04290407
	ds_load_b32 v4, v4                                         // 0000000023B0: D8D80000 04000004
	ds_load_b32 v5, v10                                        // 0000000023B8: D8D80000 0500000A
	s_wait_dscnt 0x0                                           // 0000000023C0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000023C4: 06080B04
	ds_store_b32 v10, v4                                       // 0000000023C8: D8340000 0000040A
	s_branch 65508                                             // 0000000023D0: BFA0FFE4 <ullm_cached_prefix_attn_f32_flash2_kernel+0x864>
	v_cmp_gt_u32_e64 s6, s37, v0                               // 0000000023D4: D44C0006 00020025
	v_mov_b32_e32 v4, 0xff7fffff                               // 0000000023DC: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s12, s6                                 // 0000000023E4: BE8C2006
	s_cbranch_execz 24                                         // 0000000023E8: BFA50018 <ullm_cached_prefix_attn_f32_flash2_kernel+0x94c>
	v_dual_mov_b32 v4, 0xff7fffff :: v_dual_mov_b32 v5, v14    // 0000000023EC: CA1000FF 0404010E FF7FFFFF
	v_mov_b32_e32 v6, v0                                       // 0000000023F8: 7E0C0300
	s_mov_b32 s13, 0                                           // 0000000023FC: BE8D0080
	ds_load_b32 v7, v5                                         // 000000002400: D8D80000 07000005
	v_add_nc_u32_e32 v6, s26, v6                               // 000000002408: 4A0C0C1A
	v_add_nc_u32_e32 v5, s35, v5                               // 00000000240C: 4A0A0A23
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002410: BF870002
	v_cmp_le_u32_e32 vcc_lo, s37, v6                           // 000000002414: 7C960C25
	s_wait_alu 0xfffe                                          // 000000002418: BF88FFFE
	s_or_b32 s13, vcc_lo, s13                                  // 00000000241C: 8C0D0D6A
	s_wait_dscnt 0x0                                           // 000000002420: BFC60000
	v_cmp_gt_f32_e64 s7, v7, v4                                // 000000002424: D4140007 00020907
	s_wait_alu 0xf1ff                                          // 00000000242C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002430: BF870001
	v_cndmask_b32_e64 v4, v4, v7, s7                           // 000000002434: D5010004 001E0F04
	s_wait_alu 0xfffe                                          // 00000000243C: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s13                       // 000000002440: 917E0D7E
	s_cbranch_execnz 65518                                     // 000000002444: BFA6FFEE <ullm_cached_prefix_attn_f32_flash2_kernel+0x900>
	s_or_b32 exec_lo, exec_lo, s13                             // 000000002448: 8C7E0D7E
	s_wait_alu 0xfffe                                          // 00000000244C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000002450: 8C7E0C7E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002454: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000002458: 916A217E
	s_mov_b32 s7, s26                                          // 00000000245C: BE87001A
	ds_store_b32 v10, v4                                       // 000000002460: D8340000 0000040A
	s_wait_loadcnt_dscnt 0x0                                   // 000000002468: BFC80000
	s_barrier_signal -1                                        // 00000000246C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002470: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002474: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000002480: BF88FFFE
	s_cbranch_vccz 272                                         // 000000002484: BFA30110 <ullm_cached_prefix_attn_f32_flash2_kernel+0xdc8>
	s_and_saveexec_b32 s7, s5                                  // 000000002488: BE872005
	s_cbranch_execz 53                                         // 00000000248C: BFA50035 <ullm_cached_prefix_attn_f32_flash2_kernel+0xa64>
	ds_load_b32 v4, v11                                        // 000000002490: D8D80000 0400000B
	s_wait_dscnt 0x0                                           // 000000002498: BFC60000
	v_readfirstlane_b32 s12, v4                                // 00000000249C: 7E180504
	s_cmp_gt_f32 s36, s12                                      // 0000000024A0: BF440C24
	s_cselect_b32 s12, s36, s12                                // 0000000024A4: 980C0C24
	s_wait_alu 0xfffe                                          // 0000000024A8: BF88FFFE
	s_sub_f32 s13, s36, s12                                    // 0000000024AC: A08D0C24
	v_mov_b32_e32 v5, s12                                      // 0000000024B0: 7E0A020C
	s_wait_alu 0xfffe                                          // 0000000024B4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000024B8: BF870529
	s_mul_f32 s30, s13, 0x3fb8aa3b                             // 0000000024BC: A21EFF0D 3FB8AA3B
	s_wait_alu 0xfffe                                          // 0000000024C4: BF88FFFE
	s_xor_b32 s38, s30, 0x80000000                             // 0000000024C8: 8D26FF1E 80000000
	s_rndne_f32 s39, s30                                       // 0000000024D0: BEA7631E
	s_wait_alu 0xfffe                                          // 0000000024D4: BF88FFFE
	s_fmamk_f32 s38, s13, 0x3fb8aa3b, s38                      // 0000000024D8: A326260D 3FB8AA3B
	s_cmp_nlt_f32 s13, 0xc2ce8ed0                              // 0000000024E0: BF4EFF0D C2CE8ED0
	s_sub_f32 s30, s30, s39                                    // 0000000024E8: A09E271E
	s_wait_alu 0xfffe                                          // 0000000024EC: BF88FFFE
	s_fmamk_f32 s38, s13, 0x32a5705f, s38                      // 0000000024F0: A326260D 32A5705F
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000024F8: 986A80C1
	s_cmp_ngt_f32 s13, 0x42b17218                              // 0000000024FC: BF4BFF0D 42B17218
	s_wait_alu 0xfffe                                          // 000000002504: BF88FFFE
	s_add_f32 s30, s30, s38                                    // 000000002508: A01E261E
	s_cvt_i32_f32 s38, s39                                     // 00000000250C: BEA66627
	s_wait_alu 0xfffe                                          // 000000002510: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)// 000000002514: BF8702A9
	v_s_exp_f32 s30, s30                                       // 000000002518: D680001E 0000001E
	s_wait_alu 0xf1ff                                          // 000000002520: BF88F1FF
	v_ldexp_f32 v4, s30, s38                                   // 000000002524: D71C0004 00004C1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000252C: BF8700C1
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000002530: 02080880
	s_cselect_b32 vcc_lo, -1, 0                                // 000000002534: 986A80C1
	s_cmp_nle_f32 s36, 0xff61b1e6                              // 000000002538: BF4CFF24 FF61B1E6
	s_wait_alu 0xfffe                                          // 000000002540: BF88FFFE
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo               // 000000002544: 020808FF 7F800000
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000254C: 986A80C1
	s_wait_alu 0xfffe                                          // 000000002550: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002554: BF870001
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000002558: 02080880
	ds_store_b64 v11, v[4:5] offset:1280                       // 00000000255C: D9340500 0000040B
	s_wait_alu 0xfffe                                          // 000000002564: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000002568: 8C7E077E
	s_wait_loadcnt_dscnt 0x0                                   // 00000000256C: BFC80000
	s_barrier_signal -1                                        // 000000002570: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002574: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002578: EE0AC07C 00040000 00000000
	ds_load_b32 v4, v11 offset:1284                            // 000000002584: D8D80504 0400000B
	s_wait_dscnt 0x0                                           // 00000000258C: BFC60000
	v_readfirstlane_b32 s36, v4                                // 000000002590: 7E480504
	v_mov_b32_e32 v4, 0                                        // 000000002594: 7E080280
	s_and_saveexec_b32 s7, s6                                  // 000000002598: BE872006
	s_cbranch_execz 48                                         // 00000000259C: BFA50030 <ullm_cached_prefix_attn_f32_flash2_kernel+0xb60>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, v14             // 0000000025A0: CA100080 0404010E
	v_mov_b32_e32 v6, v0                                       // 0000000025A8: 7E0C0300
	s_mov_b32 s6, 0                                            // 0000000025AC: BE860080
	ds_load_b32 v7, v5                                         // 0000000025B0: D8D80000 07000005
	s_wait_dscnt 0x0                                           // 0000000025B8: BFC60000
	v_dual_subrev_f32 v7, s36, v7 :: v_dual_add_nc_u32 v6, s26, v6// 0000000025BC: C9A00E24 07060C1A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000025C4: BF870091
	v_mul_f32_e32 v16, 0x3fb8aa3b, v7                          // 0000000025C8: 10200EFF 3FB8AA3B
	v_fma_f32 v17, 0x3fb8aa3b, v7, -v16                        // 0000000025D0: D6130011 84420EFF 3FB8AA3B
	v_rndne_f32_e32 v18, v16                                   // 0000000025DC: 7E244710
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000025E0: BF870221
	v_sub_f32_e32 v16, v16, v18                                // 0000000025E4: 08202510
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v7                   // 0000000025E8: 7C360EFF C2CE8ED0
	v_fmac_f32_e32 v17, 0x32a5705f, v7                         // 0000000025F0: 56220EFF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000025F8: BF870121
	v_add_f32_e32 v16, v16, v17                                // 0000000025FC: 06202310
	v_cvt_i32_f32_e32 v17, v18                                 // 000000002600: 7E221112
	v_exp_f32_e32 v16, v16                                     // 000000002604: 7E204B10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002608: BF8700A5
	v_ldexp_f32 v16, v16, v17                                  // 00000000260C: D71C0010 00022310
	s_wait_alu 0xfffd                                          // 000000002614: BF88FFFD
	v_cndmask_b32_e32 v16, 0, v16, vcc_lo                      // 000000002618: 02202080
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v7                   // 00000000261C: 7C3C0EFF 42B17218
	s_wait_alu 0xfffd                                          // 000000002624: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002628: BF870002
	v_cndmask_b32_e32 v7, 0x7f800000, v16, vcc_lo              // 00000000262C: 020E20FF 7F800000
	v_cmp_le_u32_e32 vcc_lo, s37, v6                           // 000000002634: 7C960C25
	ds_store_b32 v5, v7                                        // 000000002638: D8340000 00000705
	v_dual_add_f32 v4, v4, v7 :: v_dual_add_nc_u32 v5, s35, v5 // 000000002640: C9200F04 04040A23
	s_wait_alu 0xfffe                                          // 000000002648: BF88FFFE
	s_or_b32 s6, vcc_lo, s6                                    // 00000000264C: 8C06066A
	s_wait_alu 0xfffe                                          // 000000002650: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s6                        // 000000002654: 917E067E
	s_cbranch_execnz 65493                                     // 000000002658: BFA6FFD5 <ullm_cached_prefix_attn_f32_flash2_kernel+0xab0>
	s_or_b32 exec_lo, exec_lo, s6                              // 00000000265C: 8C7E067E
	s_wait_alu 0xfffe                                          // 000000002660: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000002664: 8C7E077E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002668: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 00000000266C: 916A217E
	s_mov_b32 s6, s26                                          // 000000002670: BE86001A
	ds_store_b32 v10, v4                                       // 000000002674: D8340000 0000040A
	s_wait_loadcnt_dscnt 0x0                                   // 00000000267C: BFC80000
	s_barrier_signal -1                                        // 000000002680: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002684: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002688: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000002694: BF88FFFE
	s_cbranch_vccz 169                                         // 000000002698: BFA300A9 <ullm_cached_prefix_attn_f32_flash2_kernel+0xe40>
	s_and_saveexec_b32 s6, s5                                  // 00000000269C: BE862005
	s_cbranch_execz 5                                          // 0000000026A0: BFA50005 <ullm_cached_prefix_attn_f32_flash2_kernel+0xbb8>
	ds_load_b32 v4, v11                                        // 0000000026A4: D8D80000 0400000B
	s_wait_dscnt 0x0                                           // 0000000026AC: BFC60000
	ds_store_b32 v11, v4 offset:1288                           // 0000000026B0: D8340508 0000040B
	s_wait_alu 0xfffe                                          // 0000000026B8: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000026BC: 8C7E067E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000026C0: BFC80000
	s_barrier_signal -1                                        // 0000000026C4: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000026C8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000026CC: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s6, s3                                  // 0000000026D8: BE862003
	s_wait_alu 0xfffe                                          // 0000000026DC: BF88FFFE
	s_xor_b32 s6, exec_lo, s6                                  // 0000000026E0: 8D06067E
	ds_load_b32 v5, v11 offset:1280                            // 0000000026E4: D8D80500 0500000B
	s_wait_alu 0xfffe                                          // 0000000026EC: BF88FFFE
	s_and_not1_saveexec_b32 s6, s6                             // 0000000026F0: BE863006
	s_cbranch_execz 209                                        // 0000000026F4: BFA500D1 <ullm_cached_prefix_attn_f32_flash2_kernel+0xf3c>
	v_cmp_lt_u64_e64 s7, s[8:9], 4                             // 0000000026F8: D4590007 00010808
	v_mov_b32_e32 v4, 0                                        // 000000002700: 7E080280
	s_max_u32 s8, s37, 1                                       // 000000002704: 8A888125
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000002708: 8B6A077E
	s_wait_alu 0xfffe                                          // 00000000270C: BF88FFFE
	s_cbranch_vccnz 157                                        // 000000002710: BFA4009D <ullm_cached_prefix_attn_f32_flash2_kernel+0xe88>
	s_and_b32 s7, s8, 0x7c                                     // 000000002714: 8B07FF08 0000007C
	s_mov_b32 s30, 0                                           // 00000000271C: BE9E0080
	s_movk_i32 s9, 0x400                                       // 000000002720: B0090400
	s_wait_alu 0xfffe                                          // 000000002724: BF88FFFE
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 000000002728: A98C1E1C
	s_or_b32 s38, s30, 1                                       // 00000000272C: 8C26811E
	s_mov_b32 s39, s31                                         // 000000002730: BEA7001F
	s_wait_alu 0xfffe                                          // 000000002734: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 000000002738: AA8C140C
	s_add_nc_u64 s[38:39], s[28:29], s[38:39]                  // 00000000273C: A9A6261C
	s_wait_alu 0xfffe                                          // 000000002740: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 000000002744: A98C120C
	s_mul_u64 s[38:39], s[38:39], s[20:21]                     // 000000002748: AAA61426
	s_wait_alu 0xfffe                                          // 00000000274C: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[16:17]                     // 000000002750: AA8C100C
	s_add_nc_u64 s[38:39], s[38:39], s[18:19]                  // 000000002754: A9A61226
	s_wait_alu 0xfffe                                          // 000000002758: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 00000000275C: 848C820C
	s_or_b32 s40, s30, 2                                       // 000000002760: 8C28821E
	s_mov_b32 s41, s31                                         // 000000002764: BEA9001F
	s_mul_u64 s[38:39], s[38:39], s[16:17]                     // 000000002768: AAA61026
	s_wait_dscnt 0x0                                           // 00000000276C: BFC60000
	s_wait_alu 0xfffe                                          // 000000002770: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s12                          // 000000002774: D7006A05 0000190C
	s_or_b32 s42, s30, 3                                       // 00000000277C: 8C2A831E
	s_mov_b32 s43, s31                                         // 000000002780: BEAB001F
	s_add_nc_u64 s[40:41], s[28:29], s[40:41]                  // 000000002784: A9A8281C
	s_wait_alu 0xfffd                                          // 000000002788: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s13, v13, vcc_lo             // 00000000278C: D5207C06 01AA1A0D
	s_lshl_b64 s[12:13], s[38:39], 2                           // 000000002794: 848C8226
	s_add_nc_u64 s[42:43], s[28:29], s[42:43]                  // 000000002798: A9AA2A1C
	s_wait_alu 0xfffe                                          // 00000000279C: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 0000000027A0: AAA81428
	v_add_co_u32 v16, vcc_lo, v12, s12                         // 0000000027A4: D7006A10 0000190C
	s_mul_u64 s[42:43], s[42:43], s[20:21]                     // 0000000027AC: AAAA142A
	s_wait_alu 0xfffe                                          // 0000000027B0: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[18:19]                  // 0000000027B4: A9A81228
	s_wait_alu 0xfffd                                          // 0000000027B8: BF88FFFD
	v_add_co_ci_u32_e64 v17, null, s13, v13, vcc_lo            // 0000000027BC: D5207C11 01AA1A0D
	s_add_nc_u64 s[42:43], s[42:43], s[18:19]                  // 0000000027C4: A9AA122A
	s_wait_alu 0xfffe                                          // 0000000027C8: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[16:17]                     // 0000000027CC: AAA81028
	s_clause 0x1                                               // 0000000027D0: BF850001
	global_load_b32 v7, v[5:6], off                            // 0000000027D4: EE05007C 00000007 00000005
	global_load_b32 v20, v[16:17], off                         // 0000000027E0: EE05007C 00000014 00000010
	s_mul_u64 s[42:43], s[42:43], s[16:17]                     // 0000000027EC: AAAA102A
	s_wait_alu 0xfffe                                          // 0000000027F0: BF88FFFE
	s_lshl_b64 s[38:39], s[40:41], 2                           // 0000000027F4: 84A68228
	s_lshl_b64 s[40:41], s[42:43], 2                           // 0000000027F8: 84A8822A
	s_wait_alu 0xfffe                                          // 0000000027FC: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s38                          // 000000002800: D7006A05 00004D0C
	s_wait_alu 0xfffd                                          // 000000002808: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s39, v13, vcc_lo             // 00000000280C: D5207C06 01AA1A27
	v_add_co_u32 v16, vcc_lo, v12, s40                         // 000000002814: D7006A10 0000510C
	s_wait_alu 0xfffd                                          // 00000000281C: BF88FFFD
	v_add_co_ci_u32_e64 v17, null, s41, v13, vcc_lo            // 000000002820: D5207C11 01AA1A29
	s_clause 0x1                                               // 000000002828: BF850001
	global_load_b32 v5, v[5:6], off                            // 00000000282C: EE05007C 00000005 00000005
	global_load_b32 v6, v[16:17], off                          // 000000002838: EE05007C 00000006 00000010
	v_mov_b32_e32 v16, s9                                      // 000000002844: 7E200209
	s_add_co_i32 s30, s30, 4                                   // 000000002848: 811E841E
	s_add_co_i32 s9, s9, 16                                    // 00000000284C: 81099009
	s_wait_alu 0xfffe                                          // 000000002850: BF88FFFE
	s_cmp_eq_u32 s7, s30                                       // 000000002854: BF061E07
	ds_load_b128 v[16:19], v16                                 // 000000002858: DBFC0000 10000010
	s_wait_loadcnt_dscnt 0x300                                 // 000000002860: BFC80300
	v_fmac_f32_e32 v4, v16, v7                                 // 000000002864: 56080F10
	s_wait_loadcnt 0x2                                         // 000000002868: BFC00002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000286C: BF8700A1
	v_fmac_f32_e32 v4, v17, v20                                // 000000002870: 56082911
	s_wait_loadcnt 0x1                                         // 000000002874: BFC00001
	v_fmac_f32_e32 v4, v18, v5                                 // 000000002878: 56080B12
	s_wait_loadcnt 0x0                                         // 00000000287C: BFC00000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002880: BF870001
	v_fmac_f32_e32 v4, v19, v6                                 // 000000002884: 56080D13
	s_cbranch_scc0 65446                                       // 000000002888: BFA1FFA6 <ullm_cached_prefix_attn_f32_flash2_kernel+0xc24>
	s_and_b32 s8, s8, 3                                        // 00000000288C: 8B088308
	s_wait_alu 0xfffe                                          // 000000002890: BF88FFFE
	s_cmp_eq_u32 s8, 0                                         // 000000002894: BF068008
	s_cbranch_scc0 64                                          // 000000002898: BFA10040 <ullm_cached_prefix_attn_f32_flash2_kernel+0xe9c>
	s_branch 96                                                // 00000000289C: BFA00060 <ullm_cached_prefix_attn_f32_flash2_kernel+0xf20>
	s_or_b32 exec_lo, exec_lo, s13                             // 0000000028A0: 8C7E0D7E
	s_cmp_lt_u32 s7, 4                                         // 0000000028A4: BF0A8407
	s_mov_b32 s7, s12                                          // 0000000028A8: BE87000C
	s_wait_loadcnt_dscnt 0x0                                   // 0000000028AC: BFC80000
	s_barrier_signal -1                                        // 0000000028B0: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000028B4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000028B8: EE0AC07C 00040000 00000000
	s_cbranch_scc1 65264                                       // 0000000028C4: BFA2FEF0 <ullm_cached_prefix_attn_f32_flash2_kernel+0x988>
	s_wait_alu 0xfffe                                          // 0000000028C8: BF88FFFE
	s_lshr_b32 s12, s7, 1                                      // 0000000028CC: 850C8107
	s_mov_b32 s13, exec_lo                                     // 0000000028D0: BE8D007E
	s_wait_alu 0xfffe                                          // 0000000028D4: BF88FFFE
	v_cmpx_gt_u32_e64 s12, v0                                  // 0000000028D8: D4CC007E 0002000C
	s_cbranch_execz 65519                                      // 0000000028E0: BFA5FFEF <ullm_cached_prefix_attn_f32_flash2_kernel+0xda0>
	v_lshl_add_u32 v4, s12, 2, v10                             // 0000000028E4: D6460004 0429040C
	ds_load_b32 v5, v10                                        // 0000000028EC: D8D80000 0500000A
	ds_load_b32 v4, v4                                         // 0000000028F4: D8D80000 04000004
	s_wait_dscnt 0x0                                           // 0000000028FC: BFC60000
	v_cmp_gt_f32_e32 vcc_lo, v5, v4                            // 000000002900: 7C280905
	s_wait_alu 0xfffd                                          // 000000002904: BF88FFFD
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000002908: 02080B04
	ds_store_b32 v10, v4                                       // 00000000290C: D8340000 0000040A
	s_branch 65506                                             // 000000002914: BFA0FFE2 <ullm_cached_prefix_attn_f32_flash2_kernel+0xda0>
	s_or_b32 exec_lo, exec_lo, s12                             // 000000002918: 8C7E0C7E
	s_cmp_lt_u32 s6, 4                                         // 00000000291C: BF0A8406
	s_mov_b32 s6, s7                                           // 000000002920: BE860007
	s_wait_loadcnt_dscnt 0x0                                   // 000000002924: BFC80000
	s_barrier_signal -1                                        // 000000002928: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000292C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002930: EE0AC07C 00040000 00000000
	s_cbranch_scc1 65367                                       // 00000000293C: BFA2FF57 <ullm_cached_prefix_attn_f32_flash2_kernel+0xb9c>
	s_wait_alu 0xfffe                                          // 000000002940: BF88FFFE
	s_lshr_b32 s7, s6, 1                                       // 000000002944: 85078106
	s_mov_b32 s12, exec_lo                                     // 000000002948: BE8C007E
	s_wait_alu 0xfffe                                          // 00000000294C: BF88FFFE
	v_cmpx_gt_u32_e64 s7, v0                                   // 000000002950: D4CC007E 00020007
	s_cbranch_execz 65519                                      // 000000002958: BFA5FFEF <ullm_cached_prefix_attn_f32_flash2_kernel+0xe18>
	v_lshl_add_u32 v4, s7, 2, v10                              // 00000000295C: D6460004 04290407
	ds_load_b32 v4, v4                                         // 000000002964: D8D80000 04000004
	ds_load_b32 v5, v10                                        // 00000000296C: D8D80000 0500000A
	s_wait_dscnt 0x0                                           // 000000002974: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000002978: 06080B04
	ds_store_b32 v10, v4                                       // 00000000297C: D8340000 0000040A
	s_branch 65508                                             // 000000002984: BFA0FFE4 <ullm_cached_prefix_attn_f32_flash2_kernel+0xe18>
	s_mov_b32 s7, 0                                            // 000000002988: BE870080
	s_and_b32 s8, s8, 3                                        // 00000000298C: 8B088308
	s_wait_alu 0xfffe                                          // 000000002990: BF88FFFE
	s_cmp_eq_u32 s8, 0                                         // 000000002994: BF068008
	s_cbranch_scc1 33                                          // 000000002998: BFA20021 <ullm_cached_prefix_attn_f32_flash2_kernel+0xf20>
	s_lshl_b32 s9, s7, 2                                       // 00000000299C: 84098207
	s_mov_b32 s30, s7                                          // 0000000029A0: BE9E0007
	s_wait_alu 0xfffe                                          // 0000000029A4: BF88FFFE
	s_bitset1_b32 s9, 10                                       // 0000000029A8: BE89128A
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 0000000029AC: A98C1E1C
	s_add_co_i32 s8, s8, -1                                    // 0000000029B0: 8108C108
	s_wait_alu 0xfffe                                          // 0000000029B4: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 0000000029B8: AA8C140C
	s_add_co_i32 s30, s30, 1                                   // 0000000029BC: 811E811E
	s_wait_alu 0xfffe                                          // 0000000029C0: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 0000000029C4: A98C120C
	s_wait_alu 0xfffe                                          // 0000000029C8: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[16:17]                     // 0000000029CC: AA8C100C
	s_wait_alu 0xfffe                                          // 0000000029D0: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 0000000029D4: 848C820C
	s_wait_dscnt 0x0                                           // 0000000029D8: BFC60000
	s_wait_alu 0xfffe                                          // 0000000029DC: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s12                          // 0000000029E0: D7006A05 0000190C
	s_wait_alu 0xfffd                                          // 0000000029E8: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s13, v13, vcc_lo             // 0000000029EC: D5207C06 01AA1A0D
	global_load_b32 v5, v[5:6], off                            // 0000000029F4: EE05007C 00000005 00000005
	v_mov_b32_e32 v6, s9                                       // 000000002A00: 7E0C0209
	s_add_co_i32 s9, s9, 4                                     // 000000002A04: 81098409
	s_cmp_lg_u32 s8, 0                                         // 000000002A08: BF078008
	ds_load_b32 v6, v6                                         // 000000002A0C: D8D80000 06000006
	s_wait_loadcnt_dscnt 0x0                                   // 000000002A14: BFC80000
	v_fmac_f32_e32 v4, v6, v5                                  // 000000002A18: 56080B06
	s_cbranch_scc1 65507                                       // 000000002A1C: BFA2FFE3 <ullm_cached_prefix_attn_f32_flash2_kernel+0xeac>
	s_wait_dscnt 0x0                                           // 000000002A20: BFC60000
	ds_load_b32 v5, v11 offset:1280                            // 000000002A24: D8D80500 0500000B
	s_wait_dscnt 0x0                                           // 000000002A2C: BFC60000
	v_fmac_f32_e32 v4, v9, v5                                  // 000000002A30: 56080B09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002A34: BF870001
	v_mov_b32_e32 v9, v4                                       // 000000002A38: 7E120304
	s_wait_alu 0xfffe                                          // 000000002A3C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002A40: 8C7E067E
	ds_load_b32 v4, v11 offset:1288                            // 000000002A44: D8D80508 0400000B
	s_add_nc_u64 s[28:29], s[28:29], 64                        // 000000002A4C: A99CC01C
	s_wait_loadcnt_dscnt 0x0                                   // 000000002A50: BFC80000
	s_wait_alu 0xfffe                                          // 000000002A54: BF88FFFE
	v_cmp_ge_u64_e64 s6, s[28:29], s[14:15]                    // 000000002A58: D45E0006 00001C1C
	s_barrier_signal -1                                        // 000000002A60: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002A64: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002A68: EE0AC07C 00040000 00000000
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000002A74: 8B6A067E
	v_fmac_f32_e32 v4, v15, v5                                 // 000000002A78: 56080B0F
	s_wait_alu 0xfffe                                          // 000000002A7C: BF88FFFE
	s_cbranch_vccnz 3                                          // 000000002A80: BFA40003 <ullm_cached_prefix_attn_f32_flash2_kernel+0xf90>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002A84: BF870001
	v_mov_b32_e32 v15, v4                                      // 000000002A88: 7E1E0304
	s_branch 64976                                             // 000000002A8C: BFA0FDD0 <ullm_cached_prefix_attn_f32_flash2_kernel+0x6d0>
	s_and_saveexec_b32 s3, s2                                  // 000000002A90: BE832002
	s_cbranch_execz 34                                         // 000000002A94: BFA50022 <ullm_cached_prefix_attn_f32_flash2_kernel+0x1020>
	v_div_scale_f32 v0, null, v4, v4, v9                       // 000000002A98: D6FC7C00 04260904
	s_load_b64 s[0:1], s[0:1], 0x50                            // 000000002AA0: F4002000 F8000050
	s_mul_u64 s[2:3], s[16:17], s[24:25]                       // 000000002AA8: AA821810
	s_wait_alu 0xfffe                                          // 000000002AAC: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000002AB0: 84828202
	v_rcp_f32_e32 v1, v0                                       // 000000002AB4: 7E025500
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002AB8: BF870095
	v_fma_f32 v2, -v0, v1, 1.0                                 // 000000002ABC: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 000000002AC4: 56020302
	v_div_scale_f32 v2, vcc_lo, v9, v4, v9                     // 000000002AC8: D6FC6A02 04260909
	s_wait_kmcnt 0x0                                           // 000000002AD0: BFC70000
	s_wait_alu 0xfffe                                          // 000000002AD4: BF88FFFE
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000002AD8: A9800200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002ADC: BF870091
	v_mul_f32_e32 v3, v2, v1                                   // 000000002AE0: 10060302
	v_fma_f32 v5, -v0, v3, v2                                  // 000000002AE4: D6130005 240A0700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002AEC: BF870091
	v_fmac_f32_e32 v3, v5, v1                                  // 000000002AF0: 56060305
	v_fma_f32 v0, -v0, v3, v2                                  // 000000002AF4: D6130000 240A0700
	s_wait_alu 0xfffd                                          // 000000002AFC: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002B00: BF870091
	v_div_fmas_f32 v0, v0, v1, v3                              // 000000002B04: D6370000 040E0300
	v_div_fixup_f32 v0, v0, v4, v9                             // 000000002B0C: D6270000 04260900
	global_store_b32 v8, v0, s[0:1]                            // 000000002B14: EE068000 00000000 00000008
	s_endpgm                                                   // 000000002B20: BFB00000
	s_branch 64695                                             // 000000002B24: BFA0FCB7 <ullm_cached_prefix_attn_f32_flash2_kernel+0x304>
	s_branch 64863                                             // 000000002B28: BFA0FD5F <ullm_cached_prefix_attn_f32_flash2_kernel+0x5a8>
	s_code_end                                                 // 000000002B2C: BF9F0000
	s_code_end                                                 // 000000002B30: BF9F0000
	s_code_end                                                 // 000000002B34: BF9F0000
	s_code_end                                                 // 000000002B38: BF9F0000
	s_code_end                                                 // 000000002B3C: BF9F0000
	s_code_end                                                 // 000000002B40: BF9F0000
	s_code_end                                                 // 000000002B44: BF9F0000
	s_code_end                                                 // 000000002B48: BF9F0000
	s_code_end                                                 // 000000002B4C: BF9F0000
	s_code_end                                                 // 000000002B50: BF9F0000
	s_code_end                                                 // 000000002B54: BF9F0000
	s_code_end                                                 // 000000002B58: BF9F0000
	s_code_end                                                 // 000000002B5C: BF9F0000
	s_code_end                                                 // 000000002B60: BF9F0000
	s_code_end                                                 // 000000002B64: BF9F0000
	s_code_end                                                 // 000000002B68: BF9F0000
	s_code_end                                                 // 000000002B6C: BF9F0000
	s_code_end                                                 // 000000002B70: BF9F0000
	s_code_end                                                 // 000000002B74: BF9F0000
	s_code_end                                                 // 000000002B78: BF9F0000
	s_code_end                                                 // 000000002B7C: BF9F0000
	s_code_end                                                 // 000000002B80: BF9F0000
	s_code_end                                                 // 000000002B84: BF9F0000
	s_code_end                                                 // 000000002B88: BF9F0000
	s_code_end                                                 // 000000002B8C: BF9F0000
	s_code_end                                                 // 000000002B90: BF9F0000
	s_code_end                                                 // 000000002B94: BF9F0000
	s_code_end                                                 // 000000002B98: BF9F0000
	s_code_end                                                 // 000000002B9C: BF9F0000
	s_code_end                                                 // 000000002BA0: BF9F0000
	s_code_end                                                 // 000000002BA4: BF9F0000
	s_code_end                                                 // 000000002BA8: BF9F0000
	s_code_end                                                 // 000000002BAC: BF9F0000
	s_code_end                                                 // 000000002BB0: BF9F0000
	s_code_end                                                 // 000000002BB4: BF9F0000
	s_code_end                                                 // 000000002BB8: BF9F0000
	s_code_end                                                 // 000000002BBC: BF9F0000
	s_code_end                                                 // 000000002BC0: BF9F0000
	s_code_end                                                 // 000000002BC4: BF9F0000
	s_code_end                                                 // 000000002BC8: BF9F0000
	s_code_end                                                 // 000000002BCC: BF9F0000
	s_code_end                                                 // 000000002BD0: BF9F0000
	s_code_end                                                 // 000000002BD4: BF9F0000
	s_code_end                                                 // 000000002BD8: BF9F0000
	s_code_end                                                 // 000000002BDC: BF9F0000
	s_code_end                                                 // 000000002BE0: BF9F0000
	s_code_end                                                 // 000000002BE4: BF9F0000
	s_code_end                                                 // 000000002BE8: BF9F0000
	s_code_end                                                 // 000000002BEC: BF9F0000
	s_code_end                                                 // 000000002BF0: BF9F0000
	s_code_end                                                 // 000000002BF4: BF9F0000
	s_code_end                                                 // 000000002BF8: BF9F0000
	s_code_end                                                 // 000000002BFC: BF9F0000
	s_code_end                                                 // 000000002C00: BF9F0000
	s_code_end                                                 // 000000002C04: BF9F0000
	s_code_end                                                 // 000000002C08: BF9F0000
	s_code_end                                                 // 000000002C0C: BF9F0000
	s_code_end                                                 // 000000002C10: BF9F0000
	s_code_end                                                 // 000000002C14: BF9F0000
	s_code_end                                                 // 000000002C18: BF9F0000
	s_code_end                                                 // 000000002C1C: BF9F0000
	s_code_end                                                 // 000000002C20: BF9F0000
	s_code_end                                                 // 000000002C24: BF9F0000
	s_code_end                                                 // 000000002C28: BF9F0000
	s_code_end                                                 // 000000002C2C: BF9F0000
	s_code_end                                                 // 000000002C30: BF9F0000
	s_code_end                                                 // 000000002C34: BF9F0000
	s_code_end                                                 // 000000002C38: BF9F0000
	s_code_end                                                 // 000000002C3C: BF9F0000
	s_code_end                                                 // 000000002C40: BF9F0000
	s_code_end                                                 // 000000002C44: BF9F0000
	s_code_end                                                 // 000000002C48: BF9F0000
	s_code_end                                                 // 000000002C4C: BF9F0000
	s_code_end                                                 // 000000002C50: BF9F0000
	s_code_end                                                 // 000000002C54: BF9F0000
	s_code_end                                                 // 000000002C58: BF9F0000
	s_code_end                                                 // 000000002C5C: BF9F0000
	s_code_end                                                 // 000000002C60: BF9F0000
	s_code_end                                                 // 000000002C64: BF9F0000
	s_code_end                                                 // 000000002C68: BF9F0000
	s_code_end                                                 // 000000002C6C: BF9F0000
	s_code_end                                                 // 000000002C70: BF9F0000
	s_code_end                                                 // 000000002C74: BF9F0000
	s_code_end                                                 // 000000002C78: BF9F0000
	s_code_end                                                 // 000000002C7C: BF9F0000
	s_code_end                                                 // 000000002C80: BF9F0000
	s_code_end                                                 // 000000002C84: BF9F0000
	s_code_end                                                 // 000000002C88: BF9F0000
	s_code_end                                                 // 000000002C8C: BF9F0000
	s_code_end                                                 // 000000002C90: BF9F0000
	s_code_end                                                 // 000000002C94: BF9F0000
	s_code_end                                                 // 000000002C98: BF9F0000
	s_code_end                                                 // 000000002C9C: BF9F0000
	s_code_end                                                 // 000000002CA0: BF9F0000
	s_code_end                                                 // 000000002CA4: BF9F0000
	s_code_end                                                 // 000000002CA8: BF9F0000
	s_code_end                                                 // 000000002CAC: BF9F0000
	s_code_end                                                 // 000000002CB0: BF9F0000
	s_code_end                                                 // 000000002CB4: BF9F0000
	s_code_end                                                 // 000000002CB8: BF9F0000
	s_code_end                                                 // 000000002CBC: BF9F0000
	s_code_end                                                 // 000000002CC0: BF9F0000
	s_code_end                                                 // 000000002CC4: BF9F0000
	s_code_end                                                 // 000000002CC8: BF9F0000
	s_code_end                                                 // 000000002CCC: BF9F0000
	s_code_end                                                 // 000000002CD0: BF9F0000
	s_code_end                                                 // 000000002CD4: BF9F0000
	s_code_end                                                 // 000000002CD8: BF9F0000
	s_code_end                                                 // 000000002CDC: BF9F0000
	s_code_end                                                 // 000000002CE0: BF9F0000
	s_code_end                                                 // 000000002CE4: BF9F0000
	s_code_end                                                 // 000000002CE8: BF9F0000
	s_code_end                                                 // 000000002CEC: BF9F0000
	s_code_end                                                 // 000000002CF0: BF9F0000
	s_code_end                                                 // 000000002CF4: BF9F0000
	s_code_end                                                 // 000000002CF8: BF9F0000
	s_code_end                                                 // 000000002CFC: BF9F0000
