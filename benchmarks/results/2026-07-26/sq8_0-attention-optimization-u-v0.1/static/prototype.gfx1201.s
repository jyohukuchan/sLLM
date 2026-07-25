
benchmarks/results/2026-07-26/sq8_0-attention-optimization-u-v0.1/sq8_0_r9700_attention_prototype.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000003600 <ullm_sq8_0_flash2_legacy_reference_kernel>:
	s_load_b512 s[8:23], s[0:1], 0x0                           // 000000003600: F4008200 F8000000
	s_mov_b32 s27, 0                                           // 000000003608: BE9B0080
	s_mov_b32 s24, ttmp9                                       // 00000000360C: BE980075
	s_mov_b32 s25, s27                                         // 000000003610: BE99001B
	s_wait_kmcnt 0x0                                           // 000000003614: BFC70000
	s_mul_u64 s[2:3], s[18:19], s[16:17]                       // 000000003618: AA821012
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000361C: BF870009
	v_cmp_le_u64_e64 s2, s[2:3], s[24:25]                      // 000000003620: D45B0002 00003002
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000003628: 8B6A027E
	s_cbranch_vccnz 1024                                       // 00000000362C: BFA40400 <ullm_sq8_0_flash2_legacy_reference_kernel+0x1030>
	s_clause 0x1                                               // 000000003630: BF850001
	s_load_b32 s2, s[0:1], 0x64                                // 000000003634: F4000080 F8000064
	s_load_b64 s[16:17], s[0:1], 0x40                          // 00000000363C: F4002400 F8000040
	s_wait_kmcnt 0x0                                           // 000000003644: BFC70000
	s_and_b32 s26, s2, 0xffff                                  // 000000003648: 8B1AFF02 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003650: BF870009
	v_cmp_gt_u64_e64 s2, s[16:17], s[26:27]                    // 000000003654: D45C0002 00003410
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000365C: 8B6A027E
	s_cbranch_vccnz 1011                                       // 000000003660: BFA403F3 <ullm_sq8_0_flash2_legacy_reference_kernel+0x1030>
	v_cmp_lt_u64_e64 s2, s[24:25], s[18:19]                    // 000000003664: D4590002 00002418
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000366C: 8B6A027E
	s_mov_b64 s[2:3], 0                                        // 000000003670: BE820180
	s_cbranch_vccnz 32                                         // 000000003674: BFA40020 <ullm_sq8_0_flash2_legacy_reference_kernel+0xf8>
	v_cvt_f32_u32_e32 v1, s18                                  // 000000003678: 7E020C12
	s_sub_co_i32 s3, 0, s18                                    // 00000000367C: 81831280
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003680: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000003684: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000003688: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003690: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000003694: 7E020F01
	v_readfirstlane_b32 s2, v1                                 // 000000003698: 7E040501
	s_wait_alu 0xfffe                                          // 00000000369C: BF88FFFE
	s_mul_i32 s3, s3, s2                                       // 0000000036A0: 96030203
	s_wait_alu 0xfffe                                          // 0000000036A4: BF88FFFE
	s_mul_hi_u32 s3, s2, s3                                    // 0000000036A8: 96830302
	s_wait_alu 0xfffe                                          // 0000000036AC: BF88FFFE
	s_add_co_i32 s2, s2, s3                                    // 0000000036B0: 81020302
	s_wait_alu 0xfffe                                          // 0000000036B4: BF88FFFE
	s_mul_hi_u32 s2, s24, s2                                   // 0000000036B8: 96820218
	s_wait_alu 0xfffe                                          // 0000000036BC: BF88FFFE
	s_mul_i32 s3, s2, s18                                      // 0000000036C0: 96031202
	s_add_co_i32 s4, s2, 1                                     // 0000000036C4: 81048102
	s_wait_alu 0xfffe                                          // 0000000036C8: BF88FFFE
	s_sub_co_i32 s3, s24, s3                                   // 0000000036CC: 81830318
	s_wait_alu 0xfffe                                          // 0000000036D0: BF88FFFE
	s_sub_co_i32 s5, s3, s18                                   // 0000000036D4: 81851203
	s_cmp_ge_u32 s3, s18                                       // 0000000036D8: BF091203
	s_cselect_b32 s2, s4, s2                                   // 0000000036DC: 98020204
	s_cselect_b32 s3, s5, s3                                   // 0000000036E0: 98030305
	s_wait_alu 0xfffe                                          // 0000000036E4: BF88FFFE
	s_add_co_i32 s4, s2, 1                                     // 0000000036E8: 81048102
	s_cmp_ge_u32 s3, s18                                       // 0000000036EC: BF091203
	s_mov_b32 s3, 0                                            // 0000000036F0: BE830080
	s_cselect_b32 s2, s4, s2                                   // 0000000036F4: 98020204
	s_or_b64 s[6:7], s[18:19], s[20:21]                        // 0000000036F8: 8C861412
	s_mov_b32 s6, 0                                            // 0000000036FC: BE860080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003700: BF870009
	s_cmp_lg_u64 s[6:7], 0                                     // 000000003704: BF118006
	s_cbranch_scc0 970                                         // 000000003708: BFA103CA <ullm_sq8_0_flash2_legacy_reference_kernel+0x1034>
	s_cvt_f32_u32 s4, s20                                      // 00000000370C: BE846514
	s_cvt_f32_u32 s5, s21                                      // 000000003710: BE856515
	s_sub_nc_u64 s[28:29], 0, s[20:21]                         // 000000003714: AA1C1480
	s_mov_b32 s31, s6                                          // 000000003718: BE9F0006
	s_mov_b32 s37, s6                                          // 00000000371C: BEA50006
	s_fmamk_f32 s4, s5, 0x4f800000, s4                         // 000000003720: A3040405 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003728: BF87029B
	v_s_rcp_f32 s4, s4                                         // 00000000372C: D6840004 00000004
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000003734: A204FF04 5F7FFFFC
	s_wait_alu 0xfffe                                          // 00000000373C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000003740: BF87052A
	s_mul_f32 s5, s4, 0x2f800000                               // 000000003744: A205FF04 2F800000
	s_wait_alu 0xfffe                                          // 00000000374C: BF88FFFE
	s_trunc_f32 s5, s5                                         // 000000003750: BE856205
	s_wait_alu 0xfffe                                          // 000000003754: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003758: BF8704BA
	s_fmamk_f32 s4, s5, 0xcf800000, s4                         // 00000000375C: A3040405 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000003764: BE856705
	s_wait_alu 0xfffe                                          // 000000003768: BF88FFFE
	s_cvt_u32_f32 s4, s4                                       // 00000000376C: BE846704
	s_wait_alu 0xfffe                                          // 000000003770: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003774: BF87049A
	s_mul_u64 s[34:35], s[28:29], s[4:5]                       // 000000003778: AAA2041C
	s_mul_hi_u32 s39, s4, s35                                  // 00000000377C: 96A72304
	s_mul_i32 s38, s4, s35                                     // 000000003780: 96262304
	s_mul_hi_u32 s30, s4, s34                                  // 000000003784: 969E2204
	s_mul_i32 s27, s5, s34                                     // 000000003788: 961B2205
	s_add_nc_u64 s[30:31], s[30:31], s[38:39]                  // 00000000378C: A99E261E
	s_mul_hi_u32 s7, s5, s34                                   // 000000003790: 96872205
	s_mul_hi_u32 s33, s5, s35                                  // 000000003794: 96A12305
	s_wait_alu 0xfffe                                          // 000000003798: BF88FFFE
	s_add_co_u32 s27, s30, s27                                 // 00000000379C: 801B1B1E
	s_add_co_ci_u32 s36, s31, s7                               // 0000000037A0: 8224071F
	s_mul_i32 s34, s5, s35                                     // 0000000037A4: 96222305
	s_add_co_ci_u32 s35, s33, 0                                // 0000000037A8: 82238021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000037AC: BF8704C9
	s_add_nc_u64 s[30:31], s[36:37], s[34:35]                  // 0000000037B0: A99E2224
	s_mov_b32 s35, s6                                          // 0000000037B4: BEA30006
	s_add_co_u32 s4, s4, s30                                   // 0000000037B8: 80041E04
	s_cselect_b32 s7, -1, 0                                    // 0000000037BC: 980780C1
	s_cmp_lg_u32 s7, 0                                         // 0000000037C0: BF078007
	s_add_co_ci_u32 s5, s5, s31                                // 0000000037C4: 82051F05
	s_mov_b32 s31, s6                                          // 0000000037C8: BE9F0006
	s_wait_alu 0xfffe                                          // 0000000037CC: BF88FFFE
	s_mul_u64 s[28:29], s[28:29], s[4:5]                       // 0000000037D0: AA9C041C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000037D4: BF870009
	s_mul_hi_u32 s37, s4, s29                                  // 0000000037D8: 96A51D04
	s_mul_i32 s36, s4, s29                                     // 0000000037DC: 96241D04
	s_mul_hi_u32 s30, s4, s28                                  // 0000000037E0: 969E1C04
	s_mul_i32 s27, s5, s28                                     // 0000000037E4: 961B1C05
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]                  // 0000000037E8: A99E241E
	s_mul_hi_u32 s7, s5, s28                                   // 0000000037EC: 96871C05
	s_mul_hi_u32 s33, s5, s29                                  // 0000000037F0: 96A11D05
	s_wait_alu 0xfffe                                          // 0000000037F4: BF88FFFE
	s_add_co_u32 s27, s30, s27                                 // 0000000037F8: 801B1B1E
	s_add_co_ci_u32 s34, s31, s7                               // 0000000037FC: 8222071F
	s_mul_i32 s28, s5, s29                                     // 000000003800: 961C1D05
	s_add_co_ci_u32 s29, s33, 0                                // 000000003804: 821D8021
	s_mov_b32 s31, s6                                          // 000000003808: BE9F0006
	s_add_nc_u64 s[28:29], s[34:35], s[28:29]                  // 00000000380C: A99C1C22
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003810: BF870009
	s_add_co_u32 s4, s4, s28                                   // 000000003814: 80041C04
	s_cselect_b32 s7, -1, 0                                    // 000000003818: 980780C1
	s_wait_alu 0xfffe                                          // 00000000381C: BF88FFFE
	s_mul_hi_u32 s30, s18, s4                                  // 000000003820: 969E0412
	s_cmp_lg_u32 s7, 0                                         // 000000003824: BF078007
	s_mul_hi_u32 s7, s19, s4                                   // 000000003828: 96870413
	s_add_co_ci_u32 s27, s5, s29                               // 00000000382C: 821B1D05
	s_mul_i32 s29, s19, s4                                     // 000000003830: 961D0413
	s_wait_alu 0xfffe                                          // 000000003834: BF88FFFE
	s_mul_hi_u32 s5, s18, s27                                  // 000000003838: 96851B12
	s_mul_i32 s4, s18, s27                                     // 00000000383C: 96041B12
	s_mul_hi_u32 s33, s19, s27                                 // 000000003840: 96A11B13
	s_wait_alu 0xfffe                                          // 000000003844: BF88FFFE
	s_add_nc_u64 s[4:5], s[30:31], s[4:5]                      // 000000003848: A984041E
	s_mul_i32 s28, s19, s27                                    // 00000000384C: 961C1B13
	s_wait_alu 0xfffe                                          // 000000003850: BF88FFFE
	s_add_co_u32 s4, s4, s29                                   // 000000003854: 80041D04
	s_add_co_ci_u32 s34, s5, s7                                // 000000003858: 82220705
	s_add_co_ci_u32 s29, s33, 0                                // 00000000385C: 821D8021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003860: BF8704B9
	s_add_nc_u64 s[4:5], s[34:35], s[28:29]                    // 000000003864: A9841C22
	s_wait_alu 0xfffe                                          // 000000003868: BF88FFFE
	s_mul_u64 s[28:29], s[20:21], s[4:5]                       // 00000000386C: AA9C0414
	s_sub_co_u32 s7, s18, s28                                  // 000000003870: 80871C12
	s_cselect_b32 s27, -1, 0                                   // 000000003874: 981B80C1
	s_sub_co_i32 s28, s19, s29                                 // 000000003878: 819C1D13
	s_wait_alu 0xfffe                                          // 00000000387C: BF88FFFE
	s_cmp_lg_u32 s27, 0                                        // 000000003880: BF07801B
	s_sub_co_ci_u32 s28, s28, s21                              // 000000003884: 829C151C
	s_sub_co_u32 s30, s7, s20                                  // 000000003888: 809E1407
	s_cselect_b32 s31, -1, 0                                   // 00000000388C: 981F80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000003890: BF8704A9
	s_cmp_lg_u32 s31, 0                                        // 000000003894: BF07801F
	s_sub_co_ci_u32 s28, s28, 0                                // 000000003898: 829C801C
	s_cmp_ge_u32 s28, s21                                      // 00000000389C: BF09151C
	s_cselect_b32 s33, -1, 0                                   // 0000000038A0: 982180C1
	s_cmp_ge_u32 s30, s20                                      // 0000000038A4: BF09141E
	s_add_nc_u64 s[30:31], s[4:5], 1                           // 0000000038A8: A99E8104
	s_cselect_b32 s34, -1, 0                                   // 0000000038AC: 982280C1
	s_cmp_eq_u32 s28, s21                                      // 0000000038B0: BF06151C
	s_cselect_b32 s28, s34, s33                                // 0000000038B4: 981C2122
	s_add_nc_u64 s[34:35], s[4:5], 2                           // 0000000038B8: A9A28204
	s_cmp_lg_u32 s28, 0                                        // 0000000038BC: BF07801C
	s_cselect_b32 s28, s34, s30                                // 0000000038C0: 981C1E22
	s_cselect_b32 s30, s35, s31                                // 0000000038C4: 981E1F23
	s_cmp_lg_u32 s27, 0                                        // 0000000038C8: BF07801B
	s_sub_co_ci_u32 s27, s19, s29                              // 0000000038CC: 829B1D13
	s_wait_alu 0xfffe                                          // 0000000038D0: BF88FFFE
	s_cmp_ge_u32 s27, s21                                      // 0000000038D4: BF09151B
	s_cselect_b32 s29, -1, 0                                   // 0000000038D8: 981D80C1
	s_cmp_ge_u32 s7, s20                                       // 0000000038DC: BF091407
	s_cselect_b32 s7, -1, 0                                    // 0000000038E0: 980780C1
	s_cmp_eq_u32 s27, s21                                      // 0000000038E4: BF06151B
	s_cselect_b32 s7, s7, s29                                  // 0000000038E8: 98071D07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000038EC: BF870009
	s_cmp_lg_u32 s7, 0                                         // 0000000038F0: BF078007
	s_cselect_b32 s5, s30, s5                                  // 0000000038F4: 9805051E
	s_cselect_b32 s4, s28, s4                                  // 0000000038F8: 9804041C
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 0000000038FC: 916A067E
	s_cbranch_vccnz 32                                         // 000000003900: BFA40020 <ullm_sq8_0_flash2_legacy_reference_kernel+0x384>
	v_cvt_f32_u32_e32 v1, s20                                  // 000000003904: 7E020C14
	s_sub_co_i32 s5, 0, s20                                    // 000000003908: 81851480
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000000390C: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000003910: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000003914: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000391C: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000003920: 7E020F01
	v_readfirstlane_b32 s4, v1                                 // 000000003924: 7E080501
	s_wait_alu 0xfffe                                          // 000000003928: BF88FFFE
	s_mul_i32 s5, s5, s4                                       // 00000000392C: 96050405
	s_wait_alu 0xfffe                                          // 000000003930: BF88FFFE
	s_mul_hi_u32 s5, s4, s5                                    // 000000003934: 96850504
	s_wait_alu 0xfffe                                          // 000000003938: BF88FFFE
	s_add_co_i32 s4, s4, s5                                    // 00000000393C: 81040504
	s_wait_alu 0xfffe                                          // 000000003940: BF88FFFE
	s_mul_hi_u32 s4, s18, s4                                   // 000000003944: 96840412
	s_wait_alu 0xfffe                                          // 000000003948: BF88FFFE
	s_mul_i32 s5, s4, s20                                      // 00000000394C: 96051404
	s_add_co_i32 s6, s4, 1                                     // 000000003950: 81068104
	s_wait_alu 0xfffe                                          // 000000003954: BF88FFFE
	s_sub_co_i32 s5, s18, s5                                   // 000000003958: 81850512
	s_wait_alu 0xfffe                                          // 00000000395C: BF88FFFE
	s_sub_co_i32 s7, s5, s20                                   // 000000003960: 81871405
	s_cmp_ge_u32 s5, s20                                       // 000000003964: BF091405
	s_cselect_b32 s4, s6, s4                                   // 000000003968: 98040406
	s_cselect_b32 s5, s7, s5                                   // 00000000396C: 98050507
	s_wait_alu 0xfffe                                          // 000000003970: BF88FFFE
	s_add_co_i32 s6, s4, 1                                     // 000000003974: 81068104
	s_cmp_ge_u32 s5, s20                                       // 000000003978: BF091405
	s_mov_b32 s5, 0                                            // 00000000397C: BE850080
	s_cselect_b32 s4, s6, s4                                   // 000000003980: 98040406
	s_mul_u64 s[6:7], s[2:3], s[18:19]                         // 000000003984: AA861202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000003988: BF8704C9
	s_sub_nc_u64 s[6:7], s[24:25], s[6:7]                      // 00000000398C: AA060618
	s_wait_alu 0xfffe                                          // 000000003990: BF88FFFE
	s_or_b64 s[28:29], s[6:7], s[4:5]                          // 000000003994: 8C9C0406
	s_mov_b32 s28, 0                                           // 000000003998: BE9C0080
	s_cmp_lg_u64 s[28:29], 0                                   // 00000000399C: BF11801C
	s_cbranch_scc0 805                                         // 0000000039A0: BFA10325 <ullm_sq8_0_flash2_legacy_reference_kernel+0x1038>
	s_cvt_f32_u32 s18, s4                                      // 0000000039A4: BE926504
	s_cvt_f32_u32 s19, s5                                      // 0000000039A8: BE936505
	s_sub_nc_u64 s[30:31], 0, s[4:5]                           // 0000000039AC: AA1E0480
	s_mov_b32 s35, s28                                         // 0000000039B0: BEA3001C
	s_mov_b32 s39, s28                                         // 0000000039B4: BEA7001C
	s_wait_alu 0xfffe                                          // 0000000039B8: BF88FFFE
	s_fmamk_f32 s18, s19, 0x4f800000, s18                      // 0000000039BC: A3121213 4F800000
	s_wait_alu 0xfffe                                          // 0000000039C4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000039C8: BF87029A
	v_s_rcp_f32 s18, s18                                       // 0000000039CC: D6840012 00000012
	s_mul_f32 s18, s18, 0x5f7ffffc                             // 0000000039D4: A212FF12 5F7FFFFC
	s_wait_alu 0xfffe                                          // 0000000039DC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000039E0: BF87052A
	s_mul_f32 s19, s18, 0x2f800000                             // 0000000039E4: A213FF12 2F800000
	s_wait_alu 0xfffe                                          // 0000000039EC: BF88FFFE
	s_trunc_f32 s19, s19                                       // 0000000039F0: BE936213
	s_wait_alu 0xfffe                                          // 0000000039F4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000039F8: BF8704BA
	s_fmamk_f32 s18, s19, 0xcf800000, s18                      // 0000000039FC: A3121213 CF800000
	s_cvt_u32_f32 s19, s19                                     // 000000003A04: BE936713
	s_wait_alu 0xfffe                                          // 000000003A08: BF88FFFE
	s_cvt_u32_f32 s18, s18                                     // 000000003A0C: BE926712
	s_wait_alu 0xfffe                                          // 000000003A10: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003A14: BF87049A
	s_mul_u64 s[36:37], s[30:31], s[18:19]                     // 000000003A18: AAA4121E
	s_mul_hi_u32 s41, s18, s37                                 // 000000003A1C: 96A92512
	s_mul_i32 s40, s18, s37                                    // 000000003A20: 96282512
	s_mul_hi_u32 s34, s18, s36                                 // 000000003A24: 96A22412
	s_mul_i32 s29, s19, s36                                    // 000000003A28: 961D2413
	s_add_nc_u64 s[34:35], s[34:35], s[40:41]                  // 000000003A2C: A9A22822
	s_mul_hi_u32 s27, s19, s36                                 // 000000003A30: 969B2413
	s_mul_hi_u32 s33, s19, s37                                 // 000000003A34: 96A12513
	s_add_co_u32 s29, s34, s29                                 // 000000003A38: 801D1D22
	s_wait_alu 0xfffe                                          // 000000003A3C: BF88FFFE
	s_add_co_ci_u32 s38, s35, s27                              // 000000003A40: 82261B23
	s_mul_i32 s36, s19, s37                                    // 000000003A44: 96242513
	s_add_co_ci_u32 s37, s33, 0                                // 000000003A48: 82258021
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003A4C: BF870009
	s_add_nc_u64 s[34:35], s[38:39], s[36:37]                  // 000000003A50: A9A22426
	s_mov_b32 s37, s28                                         // 000000003A54: BEA5001C
	s_add_co_u32 s18, s18, s34                                 // 000000003A58: 80122212
	s_cselect_b32 s27, -1, 0                                   // 000000003A5C: 981B80C1
	s_wait_alu 0xfffe                                          // 000000003A60: BF88FFFE
	s_cmp_lg_u32 s27, 0                                        // 000000003A64: BF07801B
	s_add_co_ci_u32 s19, s19, s35                              // 000000003A68: 82132313
	s_mov_b32 s35, s28                                         // 000000003A6C: BEA3001C
	s_wait_alu 0xfffe                                          // 000000003A70: BF88FFFE
	s_mul_u64 s[30:31], s[30:31], s[18:19]                     // 000000003A74: AA9E121E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003A78: BF870009
	s_mul_hi_u32 s39, s18, s31                                 // 000000003A7C: 96A71F12
	s_mul_i32 s38, s18, s31                                    // 000000003A80: 96261F12
	s_mul_hi_u32 s34, s18, s30                                 // 000000003A84: 96A21E12
	s_mul_i32 s29, s19, s30                                    // 000000003A88: 961D1E13
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]                  // 000000003A8C: A9A22622
	s_mul_hi_u32 s27, s19, s30                                 // 000000003A90: 969B1E13
	s_mul_hi_u32 s33, s19, s31                                 // 000000003A94: 96A11F13
	s_add_co_u32 s29, s34, s29                                 // 000000003A98: 801D1D22
	s_wait_alu 0xfffe                                          // 000000003A9C: BF88FFFE
	s_add_co_ci_u32 s36, s35, s27                              // 000000003AA0: 82241B23
	s_mul_i32 s30, s19, s31                                    // 000000003AA4: 961E1F13
	s_add_co_ci_u32 s31, s33, 0                                // 000000003AA8: 821F8021
	s_mov_b32 s35, s28                                         // 000000003AAC: BEA3001C
	s_add_nc_u64 s[30:31], s[36:37], s[30:31]                  // 000000003AB0: A99E1E24
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003AB4: BF870009
	s_add_co_u32 s18, s18, s30                                 // 000000003AB8: 80121E12
	s_cselect_b32 s27, -1, 0                                   // 000000003ABC: 981B80C1
	s_wait_alu 0xfffe                                          // 000000003AC0: BF88FFFE
	s_mul_hi_u32 s34, s6, s18                                  // 000000003AC4: 96A21206
	s_cmp_lg_u32 s27, 0                                        // 000000003AC8: BF07801B
	s_mul_hi_u32 s27, s7, s18                                  // 000000003ACC: 969B1207
	s_add_co_ci_u32 s29, s19, s31                              // 000000003AD0: 821D1F13
	s_mul_i32 s31, s7, s18                                     // 000000003AD4: 961F1207
	s_mul_hi_u32 s19, s6, s29                                  // 000000003AD8: 96931D06
	s_mul_i32 s18, s6, s29                                     // 000000003ADC: 96121D06
	s_mul_hi_u32 s33, s7, s29                                  // 000000003AE0: 96A11D07
	s_wait_alu 0xfffe                                          // 000000003AE4: BF88FFFE
	s_add_nc_u64 s[18:19], s[34:35], s[18:19]                  // 000000003AE8: A9921222
	s_mul_i32 s30, s7, s29                                     // 000000003AEC: 961E1D07
	s_wait_alu 0xfffe                                          // 000000003AF0: BF88FFFE
	s_add_co_u32 s18, s18, s31                                 // 000000003AF4: 80121F12
	s_add_co_ci_u32 s36, s19, s27                              // 000000003AF8: 82241B13
	s_add_co_ci_u32 s31, s33, 0                                // 000000003AFC: 821F8021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003B00: BF8704B9
	s_add_nc_u64 s[18:19], s[36:37], s[30:31]                  // 000000003B04: A9921E24
	s_wait_alu 0xfffe                                          // 000000003B08: BF88FFFE
	s_mul_u64 s[30:31], s[4:5], s[18:19]                       // 000000003B0C: AA9E1204
	s_sub_co_u32 s27, s6, s30                                  // 000000003B10: 809B1E06
	s_cselect_b32 s29, -1, 0                                   // 000000003B14: 981D80C1
	s_sub_co_i32 s30, s7, s31                                  // 000000003B18: 819E1F07
	s_cmp_lg_u32 s29, 0                                        // 000000003B1C: BF07801D
	s_sub_co_ci_u32 s30, s30, s5                               // 000000003B20: 829E051E
	s_wait_alu 0xfffe                                          // 000000003B24: BF88FFFE
	s_sub_co_u32 s33, s27, s4                                  // 000000003B28: 80A1041B
	s_cselect_b32 s34, -1, 0                                   // 000000003B2C: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000003B30: BF8704B9
	s_cmp_lg_u32 s34, 0                                        // 000000003B34: BF078022
	s_add_nc_u64 s[34:35], s[18:19], 1                         // 000000003B38: A9A28112
	s_sub_co_ci_u32 s30, s30, 0                                // 000000003B3C: 829E801E
	s_cmp_ge_u32 s30, s5                                       // 000000003B40: BF09051E
	s_cselect_b32 s36, -1, 0                                   // 000000003B44: 982480C1
	s_cmp_ge_u32 s33, s4                                       // 000000003B48: BF090421
	s_cselect_b32 s33, -1, 0                                   // 000000003B4C: 982180C1
	s_cmp_eq_u32 s30, s5                                       // 000000003B50: BF06051E
	s_cselect_b32 s30, s33, s36                                // 000000003B54: 981E2421
	s_add_nc_u64 s[36:37], s[18:19], 2                         // 000000003B58: A9A48212
	s_cmp_lg_u32 s30, 0                                        // 000000003B5C: BF07801E
	s_cselect_b32 s30, s36, s34                                // 000000003B60: 981E2224
	s_cselect_b32 s33, s37, s35                                // 000000003B64: 98212325
	s_cmp_lg_u32 s29, 0                                        // 000000003B68: BF07801D
	s_sub_co_ci_u32 s7, s7, s31                                // 000000003B6C: 82871F07
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003B70: BF870009
	s_cmp_ge_u32 s7, s5                                        // 000000003B74: BF090507
	s_cselect_b32 s29, -1, 0                                   // 000000003B78: 981D80C1
	s_cmp_ge_u32 s27, s4                                       // 000000003B7C: BF09041B
	s_cselect_b32 s27, -1, 0                                   // 000000003B80: 981B80C1
	s_cmp_eq_u32 s7, s5                                        // 000000003B84: BF060507
	s_wait_alu 0xfffe                                          // 000000003B88: BF88FFFE
	s_cselect_b32 s5, s27, s29                                 // 000000003B8C: 98051D1B
	s_wait_alu 0xfffe                                          // 000000003B90: BF88FFFE
	s_cmp_lg_u32 s5, 0                                         // 000000003B94: BF078005
	s_cselect_b32 s19, s33, s19                                // 000000003B98: 98131321
	s_cselect_b32 s18, s30, s18                                // 000000003B9C: 9812121E
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 000000003BA0: 916A1C7E
	s_cbranch_vccnz 29                                         // 000000003BA4: BFA4001D <ullm_sq8_0_flash2_legacy_reference_kernel+0x61c>
	v_cvt_f32_u32_e32 v1, s4                                   // 000000003BA8: 7E020C04
	s_sub_co_i32 s7, 0, s4                                     // 000000003BAC: 81870480
	s_mov_b32 s19, 0                                           // 000000003BB0: BE930080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000003BB4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000003BB8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000003BBC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003BC4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000003BC8: 7E020F01
	v_readfirstlane_b32 s5, v1                                 // 000000003BCC: 7E0A0501
	s_mul_i32 s7, s7, s5                                       // 000000003BD0: 96070507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000003BD4: BF870499
	s_mul_hi_u32 s7, s5, s7                                    // 000000003BD8: 96870705
	s_add_co_i32 s5, s5, s7                                    // 000000003BDC: 81050705
	s_wait_alu 0xfffe                                          // 000000003BE0: BF88FFFE
	s_mul_hi_u32 s5, s6, s5                                    // 000000003BE4: 96850506
	s_wait_alu 0xfffe                                          // 000000003BE8: BF88FFFE
	s_mul_i32 s7, s5, s4                                       // 000000003BEC: 96070405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003BF0: BF870009
	s_sub_co_i32 s6, s6, s7                                    // 000000003BF4: 81860706
	s_add_co_i32 s7, s5, 1                                     // 000000003BF8: 81078105
	s_sub_co_i32 s18, s6, s4                                   // 000000003BFC: 81920406
	s_cmp_ge_u32 s6, s4                                        // 000000003C00: BF090406
	s_cselect_b32 s5, s7, s5                                   // 000000003C04: 98050507
	s_wait_alu 0xfffe                                          // 000000003C08: BF88FFFE
	s_cselect_b32 s6, s18, s6                                  // 000000003C0C: 98060612
	s_add_co_i32 s7, s5, 1                                     // 000000003C10: 81078105
	s_cmp_ge_u32 s6, s4                                        // 000000003C14: BF090406
	s_cselect_b32 s18, s7, s5                                  // 000000003C18: 98120507
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v8, 2, v0       // 000000003C1C: CA220080 01080082
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]                      // 000000003C24: A982020E
	s_mov_b64 s[28:29], 0                                      // 000000003C28: BE9C0180
	s_wait_alu 0xfffe                                          // 000000003C2C: BF88FFFE
	s_add_nc_u64 s[14:15], s[2:3], 1                           // 000000003C30: A98E8102
	v_cmp_gt_u64_e64 s2, s[16:17], v[0:1]                      // 000000003C34: D45C0002 00020010
	v_cmp_le_u64_e64 s3, s[16:17], v[0:1]                      // 000000003C3C: D45B0003 00020010
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v9, v1             // 000000003C44: CA100101 04080101
	s_cmp_eq_u64 s[14:15], 0                                   // 000000003C4C: BF10800E
	s_cbranch_scc1 595                                         // 000000003C50: BFA20253 <ullm_sq8_0_flash2_legacy_reference_kernel+0xfa0>
	s_load_b32 s27, s[0:1], 0x48                               // 000000003C54: F40006C0 F8000048
	s_cmp_gt_u32 s26, 1                                        // 000000003C5C: BF08811A
	s_mul_u64 s[6:7], s[22:23], s[24:25]                       // 000000003C60: AA861816
	s_cselect_b32 s33, -1, 0                                   // 000000003C64: 982180C1
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000003C68: 84868206
	v_dual_mov_b32 v11, 0 :: v_dual_lshlrev_b32 v10, 2, v0     // 000000003C6C: CA220080 0B0A0082
	s_add_nc_u64 s[6:7], s[8:9], s[6:7]                        // 000000003C74: A9860608
	v_add_co_u32 v12, s12, s12, v8                             // 000000003C78: D7000C0C 0002100C
	v_add_co_u32 v2, s6, s6, v8                                // 000000003C80: D7000602 00021006
	v_cmp_gt_u64_e64 s4, s[22:23], v[0:1]                      // 000000003C88: D45C0004 00020016
	v_cmp_eq_u32_e64 s5, 0, v0                                 // 000000003C90: D44A0005 00020080
	s_wait_alu 0xf1ff                                          // 000000003C98: BF88F1FF
	v_add_co_ci_u32_e64 v13, null, s13, 0, s12                 // 000000003C9C: D5207C0D 0031000D
	v_add_co_ci_u32_e64 v3, null, s7, 0, s6                    // 000000003CA4: D5207C03 00190007
	v_dual_mov_b32 v15, 0 :: v_dual_add_nc_u32 v14, 0x400, v10 // 000000003CAC: CA200080 0F0E14FF 00000400
	v_mov_b32_e32 v9, 0                                        // 000000003CB8: 7E120280
	s_lshr_b32 s34, s26, 1                                     // 000000003CBC: 8522811A
	s_mov_b32 s31, 0                                           // 000000003CC0: BE9F0080
	s_lshl_b32 s35, s26, 2                                     // 000000003CC4: 8423821A
	s_lshl_b32 s36, s26, 2                                     // 000000003CC8: 8424821A
	s_mov_b32 s37, 0xff7fffff                                  // 000000003CCC: BEA500FF FF7FFFFF
	s_sub_nc_u64 s[8:9], s[14:15], s[28:29]                    // 000000003CD4: AA081C0E
	s_mov_b32 s30, s31                                         // 000000003CD8: BE9E001F
	s_wait_alu 0xfffe                                          // 000000003CDC: BF88FFFE
	v_cmp_lt_u64_e64 s6, s[8:9], 64                            // 000000003CE0: D4590006 00018008
	s_and_b32 s6, s6, exec_lo                                  // 000000003CE8: 8B067E06
	s_cselect_b32 s38, s8, 64                                  // 000000003CEC: 9826C008
	s_branch 12                                                // 000000003CF0: BFA0000C <ullm_sq8_0_flash2_legacy_reference_kernel+0x724>
	s_wait_alu 0xfffe                                          // 000000003CF4: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003CF8: 8C7E067E
	s_add_co_i32 s30, s30, 1                                   // 000000003CFC: 811E811E
	s_wait_loadcnt_dscnt 0x0                                   // 000000003D00: BFC80000
	s_wait_alu 0xfffe                                          // 000000003D04: BF88FFFE
	s_cmp_ge_u32 s30, s38                                      // 000000003D08: BF09261E
	s_barrier_signal -1                                        // 000000003D0C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003D10: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003D14: EE0AC07C 00040000 00000000
	s_cbranch_scc1 110                                         // 000000003D20: BFA2006E <ullm_sq8_0_flash2_legacy_reference_kernel+0x8dc>
	v_mov_b32_e32 v16, 0                                       // 000000003D24: 7E200280
	s_and_saveexec_b32 s7, s4                                  // 000000003D28: BE872004
	s_cbranch_execz 50                                         // 000000003D2C: BFA50032 <ullm_sq8_0_flash2_legacy_reference_kernel+0x7f8>
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 000000003D30: A98C1E1C
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v5, v3             // 000000003D34: CA100080 10040103
	s_wait_alu 0xfffe                                          // 000000003D3C: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 000000003D40: AA8C140C
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v7, v1             // 000000003D44: CA100102 04060101
	s_wait_alu 0xfffe                                          // 000000003D4C: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 000000003D50: A98C120C
	v_mov_b32_e32 v6, v0                                       // 000000003D54: 7E0C0300
	s_wait_alu 0xfffe                                          // 000000003D58: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[22:23]                     // 000000003D5C: AA8C160C
	s_mov_b32 s39, 0                                           // 000000003D60: BEA70080
	s_wait_alu 0xfffe                                          // 000000003D64: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 000000003D68: 848C820C
	s_wait_alu 0xfffe                                          // 000000003D6C: BF88FFFE
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]                  // 000000003D70: A98C0C0A
	v_lshlrev_b64_e32 v[17:18], 2, v[6:7]                      // 000000003D74: 3E220C82
	s_wait_alu 0xfffe                                          // 000000003D78: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003D7C: BF870121
	v_add_co_u32 v17, vcc_lo, s12, v17                         // 000000003D80: D7006A11 0002220C
	s_wait_alu 0xfffd                                          // 000000003D88: BF88FFFD
	v_add_co_ci_u32_e64 v18, null, s13, v18, vcc_lo            // 000000003D8C: D5207C12 01AA240D
	v_add_co_u32 v6, vcc_lo, v6, s26                           // 000000003D94: D7006A06 00003506
	global_load_b32 v19, v[4:5], off                           // 000000003D9C: EE05007C 00000013 00000004
	global_load_b32 v17, v[17:18], off                         // 000000003DA8: EE05007C 00000011 00000011
	s_wait_alu 0xfffd                                          // 000000003DB4: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo                // 000000003DB8: D5207C07 01AA0E80
	v_add_co_u32 v4, s6, v4, s35                               // 000000003DC0: D7000604 00004704
	s_wait_alu 0xf1ff                                          // 000000003DC8: BF88F1FF
	v_add_co_ci_u32_e64 v5, null, 0, v5, s6                    // 000000003DCC: D5207C05 001A0A80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003DD4: BF870003
	v_cmp_le_u64_e32 vcc_lo, s[22:23], v[6:7]                  // 000000003DD8: 7CB60C16
	s_or_b32 s39, vcc_lo, s39                                  // 000000003DDC: 8C27276A
	s_wait_loadcnt 0x0                                         // 000000003DE0: BFC00000
	v_fmac_f32_e32 v16, v19, v17                               // 000000003DE4: 56202313
	s_wait_alu 0xfffe                                          // 000000003DE8: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s39                       // 000000003DEC: 917E277E
	s_cbranch_execnz 65504                                     // 000000003DF0: BFA6FFE0 <ullm_sq8_0_flash2_legacy_reference_kernel+0x774>
	s_or_b32 exec_lo, exec_lo, s39                             // 000000003DF4: 8C7E277E
	s_wait_alu 0xfffe                                          // 000000003DF8: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003DFC: 8C7E077E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003E00: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000003E04: 916A217E
	s_mov_b32 s6, s34                                          // 000000003E08: BE860022
	ds_store_b32 v10, v16                                      // 000000003E0C: D8340000 0000100A
	s_wait_dscnt 0x0                                           // 000000003E14: BFC60000
	s_barrier_signal -1                                        // 000000003E18: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003E1C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003E20: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000003E2C: BF88FFFE
	s_cbranch_vccz 26                                          // 000000003E30: BFA3001A <ullm_sq8_0_flash2_legacy_reference_kernel+0x89c>
	s_and_saveexec_b32 s6, s5                                  // 000000003E34: BE862005
	s_cbranch_execz 65454                                      // 000000003E38: BFA5FFAE <ullm_sq8_0_flash2_legacy_reference_kernel+0x6f4>
	ds_load_b32 v4, v11                                        // 000000003E3C: D8D80000 0400000B
	s_lshl_b32 s7, s30, 2                                      // 000000003E44: 8407821E
	s_wait_dscnt 0x0                                           // 000000003E48: BFC60000
	s_wait_kmcnt 0x0                                           // 000000003E4C: BFC70000
	s_wait_alu 0xfffe                                          // 000000003E50: BF88FFFE
	v_dual_mov_b32 v5, s7 :: v_dual_mul_f32 v4, s27, v4        // 000000003E54: CA060007 0504081B
	ds_store_b32 v5, v4 offset:1024                            // 000000003E5C: D8340400 00000405
	s_branch 65443                                             // 000000003E64: BFA0FFA3 <ullm_sq8_0_flash2_legacy_reference_kernel+0x6f4>
	s_wait_alu 0xfffe                                          // 000000003E68: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003E6C: 8C7E077E
	s_lshr_b32 s7, s6, 1                                       // 000000003E70: 85078106
	s_cmp_gt_u32 s6, 1                                         // 000000003E74: BF088106
	s_wait_alu 0xfffe                                          // 000000003E78: BF88FFFE
	s_mov_b32 s6, s7                                           // 000000003E7C: BE860007
	s_wait_loadcnt_dscnt 0x0                                   // 000000003E80: BFC80000
	s_barrier_signal -1                                        // 000000003E84: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003E88: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003E8C: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65510                                       // 000000003E98: BFA1FFE6 <ullm_sq8_0_flash2_legacy_reference_kernel+0x834>
	s_mov_b32 s7, exec_lo                                      // 000000003E9C: BE87007E
	s_wait_alu 0xfffe                                          // 000000003EA0: BF88FFFE
	v_cmpx_gt_u32_e64 s6, v0                                   // 000000003EA4: D4CC007E 00020006
	s_cbranch_execz 65518                                      // 000000003EAC: BFA5FFEE <ullm_sq8_0_flash2_legacy_reference_kernel+0x868>
	v_lshl_add_u32 v4, s6, 2, v10                              // 000000003EB0: D6460004 04290406
	ds_load_b32 v4, v4                                         // 000000003EB8: D8D80000 04000004
	ds_load_b32 v5, v10                                        // 000000003EC0: D8D80000 0500000A
	s_wait_dscnt 0x0                                           // 000000003EC8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000003ECC: 06080B04
	ds_store_b32 v10, v4                                       // 000000003ED0: D8340000 0000040A
	s_branch 65507                                             // 000000003ED8: BFA0FFE3 <ullm_sq8_0_flash2_legacy_reference_kernel+0x868>
	v_cmp_gt_u32_e64 s6, s38, v0                               // 000000003EDC: D44C0006 00020026
	v_mov_b32_e32 v4, 0xff7fffff                               // 000000003EE4: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s12, s6                                 // 000000003EEC: BE8C2006
	s_cbranch_execz 24                                         // 000000003EF0: BFA50018 <ullm_sq8_0_flash2_legacy_reference_kernel+0x954>
	v_dual_mov_b32 v4, 0xff7fffff :: v_dual_mov_b32 v5, v14    // 000000003EF4: CA1000FF 0404010E FF7FFFFF
	v_mov_b32_e32 v6, v0                                       // 000000003F00: 7E0C0300
	s_mov_b32 s13, 0                                           // 000000003F04: BE8D0080
	ds_load_b32 v7, v5                                         // 000000003F08: D8D80000 07000005
	v_add_nc_u32_e32 v6, s26, v6                               // 000000003F10: 4A0C0C1A
	v_add_nc_u32_e32 v5, s36, v5                               // 000000003F14: 4A0A0A24
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003F18: BF870002
	v_cmp_le_u32_e32 vcc_lo, s38, v6                           // 000000003F1C: 7C960C26
	s_wait_alu 0xfffe                                          // 000000003F20: BF88FFFE
	s_or_b32 s13, vcc_lo, s13                                  // 000000003F24: 8C0D0D6A
	s_wait_dscnt 0x0                                           // 000000003F28: BFC60000
	v_cmp_gt_f32_e64 s7, v7, v4                                // 000000003F2C: D4140007 00020907
	s_wait_alu 0xf1ff                                          // 000000003F34: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003F38: BF870001
	v_cndmask_b32_e64 v4, v4, v7, s7                           // 000000003F3C: D5010004 001E0F04
	s_wait_alu 0xfffe                                          // 000000003F44: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s13                       // 000000003F48: 917E0D7E
	s_cbranch_execnz 65518                                     // 000000003F4C: BFA6FFEE <ullm_sq8_0_flash2_legacy_reference_kernel+0x908>
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003F50: 8C7E0D7E
	s_wait_alu 0xfffe                                          // 000000003F54: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003F58: 8C7E0C7E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003F5C: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000003F60: 916A217E
	s_mov_b32 s7, s34                                          // 000000003F64: BE870022
	ds_store_b32 v10, v4                                       // 000000003F68: D8340000 0000040A
	s_wait_loadcnt_dscnt 0x0                                   // 000000003F70: BFC80000
	s_barrier_signal -1                                        // 000000003F74: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003F78: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003F7C: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000003F88: BF88FFFE
	s_cbranch_vccz 275                                         // 000000003F8C: BFA30113 <ullm_sq8_0_flash2_legacy_reference_kernel+0xddc>
	s_and_saveexec_b32 s7, s5                                  // 000000003F90: BE872005
	s_cbranch_execz 53                                         // 000000003F94: BFA50035 <ullm_sq8_0_flash2_legacy_reference_kernel+0xa6c>
	ds_load_b32 v6, v11                                        // 000000003F98: D8D80000 0600000B
	s_wait_dscnt 0x0                                           // 000000003FA0: BFC60000
	v_readfirstlane_b32 s12, v6                                // 000000003FA4: 7E180506
	s_cmp_gt_f32 s37, s12                                      // 000000003FA8: BF440C25
	s_cselect_b32 s12, s37, s12                                // 000000003FAC: 980C0C25
	s_wait_alu 0xfffe                                          // 000000003FB0: BF88FFFE
	s_sub_f32 s13, s37, s12                                    // 000000003FB4: A08D0C25
	v_mov_b32_e32 v5, s12                                      // 000000003FB8: 7E0A020C
	s_wait_alu 0xfffe                                          // 000000003FBC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000003FC0: BF870529
	s_mul_f32 s30, s13, 0x3fb8aa3b                             // 000000003FC4: A21EFF0D 3FB8AA3B
	s_wait_alu 0xfffe                                          // 000000003FCC: BF88FFFE
	s_xor_b32 s39, s30, 0x80000000                             // 000000003FD0: 8D27FF1E 80000000
	s_rndne_f32 s40, s30                                       // 000000003FD8: BEA8631E
	s_wait_alu 0xfffe                                          // 000000003FDC: BF88FFFE
	s_fmamk_f32 s39, s13, 0x3fb8aa3b, s39                      // 000000003FE0: A327270D 3FB8AA3B
	s_cmp_nlt_f32 s13, 0xc2ce8ed0                              // 000000003FE8: BF4EFF0D C2CE8ED0
	s_sub_f32 s30, s30, s40                                    // 000000003FF0: A09E281E
	s_wait_alu 0xfffe                                          // 000000003FF4: BF88FFFE
	s_fmamk_f32 s39, s13, 0x32a5705f, s39                      // 000000003FF8: A327270D 32A5705F
	s_cselect_b32 vcc_lo, -1, 0                                // 000000004000: 986A80C1
	s_cmp_ngt_f32 s13, 0x42b17218                              // 000000004004: BF4BFF0D 42B17218
	s_wait_alu 0xfffe                                          // 00000000400C: BF88FFFE
	s_add_f32 s30, s30, s39                                    // 000000004010: A01E271E
	s_cvt_i32_f32 s39, s40                                     // 000000004014: BEA76628
	s_wait_alu 0xfffe                                          // 000000004018: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)// 00000000401C: BF8702A9
	v_s_exp_f32 s30, s30                                       // 000000004020: D680001E 0000001E
	s_wait_alu 0xf1ff                                          // 000000004028: BF88F1FF
	v_ldexp_f32 v4, s30, s39                                   // 00000000402C: D71C0004 00004E1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000004034: BF8700C1
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000004038: 02080880
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000403C: 986A80C1
	s_cmp_nle_f32 s37, 0xff61b1e6                              // 000000004040: BF4CFF25 FF61B1E6
	s_wait_alu 0xfffe                                          // 000000004048: BF88FFFE
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo               // 00000000404C: 020808FF 7F800000
	s_cselect_b32 vcc_lo, -1, 0                                // 000000004054: 986A80C1
	s_wait_alu 0xfffe                                          // 000000004058: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000405C: BF870001
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000004060: 02080880
	ds_store_b96 v11, v[4:6] offset:1280                       // 000000004064: DB780500 0000040B
	s_wait_alu 0xfffe                                          // 00000000406C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000004070: 8C7E077E
	s_wait_loadcnt_dscnt 0x0                                   // 000000004074: BFC80000
	s_barrier_signal -1                                        // 000000004078: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000407C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004080: EE0AC07C 00040000 00000000
	ds_load_b32 v4, v11 offset:1284                            // 00000000408C: D8D80504 0400000B
	s_wait_dscnt 0x0                                           // 000000004094: BFC60000
	v_readfirstlane_b32 s37, v4                                // 000000004098: 7E4A0504
	v_mov_b32_e32 v4, 0                                        // 00000000409C: 7E080280
	s_and_saveexec_b32 s7, s6                                  // 0000000040A0: BE872006
	s_cbranch_execz 48                                         // 0000000040A4: BFA50030 <ullm_sq8_0_flash2_legacy_reference_kernel+0xb68>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, v14             // 0000000040A8: CA100080 0404010E
	v_mov_b32_e32 v6, v0                                       // 0000000040B0: 7E0C0300
	s_mov_b32 s6, 0                                            // 0000000040B4: BE860080
	ds_load_b32 v7, v5                                         // 0000000040B8: D8D80000 07000005
	s_wait_dscnt 0x0                                           // 0000000040C0: BFC60000
	v_dual_subrev_f32 v7, s37, v7 :: v_dual_add_nc_u32 v6, s26, v6// 0000000040C4: C9A00E25 07060C1A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000040CC: BF870091
	v_mul_f32_e32 v16, 0x3fb8aa3b, v7                          // 0000000040D0: 10200EFF 3FB8AA3B
	v_fma_f32 v17, 0x3fb8aa3b, v7, -v16                        // 0000000040D8: D6130011 84420EFF 3FB8AA3B
	v_rndne_f32_e32 v18, v16                                   // 0000000040E4: 7E244710
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000040E8: BF870221
	v_sub_f32_e32 v16, v16, v18                                // 0000000040EC: 08202510
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v7                   // 0000000040F0: 7C360EFF C2CE8ED0
	v_fmac_f32_e32 v17, 0x32a5705f, v7                         // 0000000040F8: 56220EFF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004100: BF870121
	v_add_f32_e32 v16, v16, v17                                // 000000004104: 06202310
	v_cvt_i32_f32_e32 v17, v18                                 // 000000004108: 7E221112
	v_exp_f32_e32 v16, v16                                     // 00000000410C: 7E204B10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004110: BF8700A5
	v_ldexp_f32 v16, v16, v17                                  // 000000004114: D71C0010 00022310
	s_wait_alu 0xfffd                                          // 00000000411C: BF88FFFD
	v_cndmask_b32_e32 v16, 0, v16, vcc_lo                      // 000000004120: 02202080
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v7                   // 000000004124: 7C3C0EFF 42B17218
	s_wait_alu 0xfffd                                          // 00000000412C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004130: BF870002
	v_cndmask_b32_e32 v7, 0x7f800000, v16, vcc_lo              // 000000004134: 020E20FF 7F800000
	v_cmp_le_u32_e32 vcc_lo, s38, v6                           // 00000000413C: 7C960C26
	ds_store_b32 v5, v7                                        // 000000004140: D8340000 00000705
	v_dual_add_f32 v4, v4, v7 :: v_dual_add_nc_u32 v5, s36, v5 // 000000004148: C9200F04 04040A24
	s_wait_alu 0xfffe                                          // 000000004150: BF88FFFE
	s_or_b32 s6, vcc_lo, s6                                    // 000000004154: 8C06066A
	s_wait_alu 0xfffe                                          // 000000004158: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s6                        // 00000000415C: 917E067E
	s_cbranch_execnz 65493                                     // 000000004160: BFA6FFD5 <ullm_sq8_0_flash2_legacy_reference_kernel+0xab8>
	s_or_b32 exec_lo, exec_lo, s6                              // 000000004164: 8C7E067E
	s_wait_alu 0xfffe                                          // 000000004168: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000416C: 8C7E077E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004170: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000004174: 916A217E
	s_mov_b32 s6, s34                                          // 000000004178: BE860022
	ds_store_b32 v10, v4                                       // 00000000417C: D8340000 0000040A
	s_wait_loadcnt_dscnt 0x0                                   // 000000004184: BFC80000
	s_barrier_signal -1                                        // 000000004188: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000418C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004190: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 00000000419C: BF88FFFE
	s_cbranch_vccz 173                                         // 0000000041A0: BFA300AD <ullm_sq8_0_flash2_legacy_reference_kernel+0xe58>
	s_and_saveexec_b32 s6, s5                                  // 0000000041A4: BE862005
	s_cbranch_execz 5                                          // 0000000041A8: BFA50005 <ullm_sq8_0_flash2_legacy_reference_kernel+0xbc0>
	ds_load_b32 v4, v11                                        // 0000000041AC: D8D80000 0400000B
	s_wait_dscnt 0x0                                           // 0000000041B4: BFC60000
	ds_store_b32 v11, v4 offset:1292                           // 0000000041B8: D834050C 0000040B
	s_wait_alu 0xfffe                                          // 0000000041C0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000041C4: 8C7E067E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000041C8: BFC80000
	s_barrier_signal -1                                        // 0000000041CC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000041D0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000041D4: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s6, s3                                  // 0000000041E0: BE862003
	s_wait_alu 0xfffe                                          // 0000000041E4: BF88FFFE
	s_xor_b32 s6, exec_lo, s6                                  // 0000000041E8: 8D06067E
	ds_load_b32 v5, v11 offset:1280                            // 0000000041EC: D8D80500 0500000B
	s_wait_alu 0xfffe                                          // 0000000041F4: BF88FFFE
	s_and_not1_saveexec_b32 s6, s6                             // 0000000041F8: BE863006
	s_cbranch_execz 211                                        // 0000000041FC: BFA500D3 <ullm_sq8_0_flash2_legacy_reference_kernel+0xf4c>
	v_cmp_lt_u64_e64 s7, s[8:9], 4                             // 000000004200: D4590007 00010808
	v_mov_b32_e32 v4, 0                                        // 000000004208: 7E080280
	s_max_u32 s8, s38, 1                                       // 00000000420C: 8A888126
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000004210: 8B6A077E
	s_wait_alu 0xfffe                                          // 000000004214: BF88FFFE
	s_cbranch_vccnz 159                                        // 000000004218: BFA4009F <ullm_sq8_0_flash2_legacy_reference_kernel+0xe98>
	s_and_b32 s7, s8, 0x7c                                     // 00000000421C: 8B07FF08 0000007C
	s_mov_b32 s30, 0                                           // 000000004224: BE9E0080
	s_movk_i32 s9, 0x400                                       // 000000004228: B0090400
	s_wait_alu 0xfffe                                          // 00000000422C: BF88FFFE
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 000000004230: A98C1E1C
	s_or_b32 s38, s30, 1                                       // 000000004234: 8C26811E
	s_mov_b32 s39, s31                                         // 000000004238: BEA7001F
	s_wait_alu 0xfffe                                          // 00000000423C: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 000000004240: AA8C140C
	s_add_nc_u64 s[38:39], s[28:29], s[38:39]                  // 000000004244: A9A6261C
	s_wait_alu 0xfffe                                          // 000000004248: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 00000000424C: A98C120C
	s_mul_u64 s[38:39], s[38:39], s[20:21]                     // 000000004250: AAA61426
	s_wait_alu 0xfffe                                          // 000000004254: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[16:17]                     // 000000004258: AA8C100C
	s_add_nc_u64 s[38:39], s[38:39], s[18:19]                  // 00000000425C: A9A61226
	s_wait_alu 0xfffe                                          // 000000004260: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 000000004264: 848C820C
	s_or_b32 s40, s30, 2                                       // 000000004268: 8C28821E
	s_mov_b32 s41, s31                                         // 00000000426C: BEA9001F
	s_mul_u64 s[38:39], s[38:39], s[16:17]                     // 000000004270: AAA61026
	s_wait_dscnt 0x0                                           // 000000004274: BFC60000
	s_wait_alu 0xfffe                                          // 000000004278: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s12                          // 00000000427C: D7006A05 0000190C
	s_or_b32 s42, s30, 3                                       // 000000004284: 8C2A831E
	s_mov_b32 s43, s31                                         // 000000004288: BEAB001F
	s_add_nc_u64 s[40:41], s[28:29], s[40:41]                  // 00000000428C: A9A8281C
	s_wait_alu 0xfffd                                          // 000000004290: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s13, v13, vcc_lo             // 000000004294: D5207C06 01AA1A0D
	s_lshl_b64 s[12:13], s[38:39], 2                           // 00000000429C: 848C8226
	s_add_nc_u64 s[42:43], s[28:29], s[42:43]                  // 0000000042A0: A9AA2A1C
	s_wait_alu 0xfffe                                          // 0000000042A4: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 0000000042A8: AAA81428
	v_add_co_u32 v16, vcc_lo, v12, s12                         // 0000000042AC: D7006A10 0000190C
	s_mul_u64 s[42:43], s[42:43], s[20:21]                     // 0000000042B4: AAAA142A
	s_wait_alu 0xfffe                                          // 0000000042B8: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[18:19]                  // 0000000042BC: A9A81228
	s_wait_alu 0xfffd                                          // 0000000042C0: BF88FFFD
	v_add_co_ci_u32_e64 v17, null, s13, v13, vcc_lo            // 0000000042C4: D5207C11 01AA1A0D
	s_add_nc_u64 s[42:43], s[42:43], s[18:19]                  // 0000000042CC: A9AA122A
	s_wait_alu 0xfffe                                          // 0000000042D0: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[16:17]                     // 0000000042D4: AAA81028
	s_clause 0x1                                               // 0000000042D8: BF850001
	global_load_b32 v7, v[5:6], off                            // 0000000042DC: EE05007C 00000007 00000005
	global_load_b32 v20, v[16:17], off                         // 0000000042E8: EE05007C 00000014 00000010
	s_mul_u64 s[42:43], s[42:43], s[16:17]                     // 0000000042F4: AAAA102A
	s_wait_alu 0xfffe                                          // 0000000042F8: BF88FFFE
	s_lshl_b64 s[38:39], s[40:41], 2                           // 0000000042FC: 84A68228
	s_lshl_b64 s[40:41], s[42:43], 2                           // 000000004300: 84A8822A
	s_wait_alu 0xfffe                                          // 000000004304: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s38                          // 000000004308: D7006A05 00004D0C
	s_wait_alu 0xfffd                                          // 000000004310: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s39, v13, vcc_lo             // 000000004314: D5207C06 01AA1A27
	v_add_co_u32 v16, vcc_lo, v12, s40                         // 00000000431C: D7006A10 0000510C
	s_wait_alu 0xfffd                                          // 000000004324: BF88FFFD
	v_add_co_ci_u32_e64 v17, null, s41, v13, vcc_lo            // 000000004328: D5207C11 01AA1A29
	s_clause 0x1                                               // 000000004330: BF850001
	global_load_b32 v5, v[5:6], off                            // 000000004334: EE05007C 00000005 00000005
	global_load_b32 v6, v[16:17], off                          // 000000004340: EE05007C 00000006 00000010
	v_mov_b32_e32 v16, s9                                      // 00000000434C: 7E200209
	s_add_co_i32 s30, s30, 4                                   // 000000004350: 811E841E
	s_add_co_i32 s9, s9, 16                                    // 000000004354: 81099009
	s_wait_alu 0xfffe                                          // 000000004358: BF88FFFE
	s_cmp_eq_u32 s7, s30                                       // 00000000435C: BF061E07
	ds_load_b128 v[16:19], v16                                 // 000000004360: DBFC0000 10000010
	s_wait_loadcnt_dscnt 0x300                                 // 000000004368: BFC80300
	v_fmac_f32_e32 v4, v16, v7                                 // 00000000436C: 56080F10
	s_wait_loadcnt 0x2                                         // 000000004370: BFC00002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004374: BF8700A1
	v_fmac_f32_e32 v4, v17, v20                                // 000000004378: 56082911
	s_wait_loadcnt 0x1                                         // 00000000437C: BFC00001
	v_fmac_f32_e32 v4, v18, v5                                 // 000000004380: 56080B12
	s_wait_loadcnt 0x0                                         // 000000004384: BFC00000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004388: BF870001
	v_fmac_f32_e32 v4, v19, v6                                 // 00000000438C: 56080D13
	s_cbranch_scc0 65446                                       // 000000004390: BFA1FFA6 <ullm_sq8_0_flash2_legacy_reference_kernel+0xc2c>
	s_and_b32 s8, s8, 3                                        // 000000004394: 8B088308
	s_wait_alu 0xfffe                                          // 000000004398: BF88FFFE
	s_cmp_eq_u32 s8, 0                                         // 00000000439C: BF068008
	s_cbranch_scc0 66                                          // 0000000043A0: BFA10042 <ullm_sq8_0_flash2_legacy_reference_kernel+0xeac>
	s_branch 98                                                // 0000000043A4: BFA00062 <ullm_sq8_0_flash2_legacy_reference_kernel+0xf30>
	s_wait_alu 0xfffe                                          // 0000000043A8: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s12                             // 0000000043AC: 8C7E0C7E
	s_lshr_b32 s12, s7, 1                                      // 0000000043B0: 850C8107
	s_cmp_gt_u32 s7, 1                                         // 0000000043B4: BF088107
	s_wait_alu 0xfffe                                          // 0000000043B8: BF88FFFE
	s_mov_b32 s7, s12                                          // 0000000043BC: BE87000C
	s_wait_loadcnt_dscnt 0x0                                   // 0000000043C0: BFC80000
	s_barrier_signal -1                                        // 0000000043C4: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000043C8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000043CC: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65261                                       // 0000000043D8: BFA1FEED <ullm_sq8_0_flash2_legacy_reference_kernel+0x990>
	s_mov_b32 s12, exec_lo                                     // 0000000043DC: BE8C007E
	s_wait_alu 0xfffe                                          // 0000000043E0: BF88FFFE
	v_cmpx_gt_u32_e64 s7, v0                                   // 0000000043E4: D4CC007E 00020007
	s_cbranch_execz 65518                                      // 0000000043EC: BFA5FFEE <ullm_sq8_0_flash2_legacy_reference_kernel+0xda8>
	v_lshl_add_u32 v4, s7, 2, v10                              // 0000000043F0: D6460004 04290407
	ds_load_b32 v5, v10                                        // 0000000043F8: D8D80000 0500000A
	ds_load_b32 v4, v4                                         // 000000004400: D8D80000 04000004
	s_wait_dscnt 0x0                                           // 000000004408: BFC60000
	v_cmp_gt_f32_e32 vcc_lo, v5, v4                            // 00000000440C: 7C280905
	s_wait_alu 0xfffd                                          // 000000004410: BF88FFFD
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000004414: 02080B04
	ds_store_b32 v10, v4                                       // 000000004418: D8340000 0000040A
	s_branch 65505                                             // 000000004420: BFA0FFE1 <ullm_sq8_0_flash2_legacy_reference_kernel+0xda8>
	s_wait_alu 0xfffe                                          // 000000004424: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s7                              // 000000004428: 8C7E077E
	s_lshr_b32 s7, s6, 1                                       // 00000000442C: 85078106
	s_cmp_gt_u32 s6, 1                                         // 000000004430: BF088106
	s_wait_alu 0xfffe                                          // 000000004434: BF88FFFE
	s_mov_b32 s6, s7                                           // 000000004438: BE860007
	s_wait_loadcnt_dscnt 0x0                                   // 00000000443C: BFC80000
	s_barrier_signal -1                                        // 000000004440: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004444: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004448: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65363                                       // 000000004454: BFA1FF53 <ullm_sq8_0_flash2_legacy_reference_kernel+0xba4>
	s_mov_b32 s7, exec_lo                                      // 000000004458: BE87007E
	s_wait_alu 0xfffe                                          // 00000000445C: BF88FFFE
	v_cmpx_gt_u32_e64 s6, v0                                   // 000000004460: D4CC007E 00020006
	s_cbranch_execz 65518                                      // 000000004468: BFA5FFEE <ullm_sq8_0_flash2_legacy_reference_kernel+0xe24>
	v_lshl_add_u32 v4, s6, 2, v10                              // 00000000446C: D6460004 04290406
	ds_load_b32 v4, v4                                         // 000000004474: D8D80000 04000004
	ds_load_b32 v5, v10                                        // 00000000447C: D8D80000 0500000A
	s_wait_dscnt 0x0                                           // 000000004484: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000004488: 06080B04
	ds_store_b32 v10, v4                                       // 00000000448C: D8340000 0000040A
	s_branch 65507                                             // 000000004494: BFA0FFE3 <ullm_sq8_0_flash2_legacy_reference_kernel+0xe24>
	s_mov_b32 s7, 0                                            // 000000004498: BE870080
	s_and_b32 s8, s8, 3                                        // 00000000449C: 8B088308
	s_wait_alu 0xfffe                                          // 0000000044A0: BF88FFFE
	s_cmp_eq_u32 s8, 0                                         // 0000000044A4: BF068008
	s_cbranch_scc1 33                                          // 0000000044A8: BFA20021 <ullm_sq8_0_flash2_legacy_reference_kernel+0xf30>
	s_lshl_b32 s9, s7, 2                                       // 0000000044AC: 84098207
	s_mov_b32 s30, s7                                          // 0000000044B0: BE9E0007
	s_wait_alu 0xfffe                                          // 0000000044B4: BF88FFFE
	s_bitset1_b32 s9, 10                                       // 0000000044B8: BE89128A
	s_add_nc_u64 s[12:13], s[28:29], s[30:31]                  // 0000000044BC: A98C1E1C
	s_add_co_i32 s8, s8, -1                                    // 0000000044C0: 8108C108
	s_wait_alu 0xfffe                                          // 0000000044C4: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[20:21]                     // 0000000044C8: AA8C140C
	s_add_co_i32 s30, s30, 1                                   // 0000000044CC: 811E811E
	s_wait_alu 0xfffe                                          // 0000000044D0: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[18:19]                  // 0000000044D4: A98C120C
	s_wait_alu 0xfffe                                          // 0000000044D8: BF88FFFE
	s_mul_u64 s[12:13], s[12:13], s[16:17]                     // 0000000044DC: AA8C100C
	s_wait_alu 0xfffe                                          // 0000000044E0: BF88FFFE
	s_lshl_b64 s[12:13], s[12:13], 2                           // 0000000044E4: 848C820C
	s_wait_dscnt 0x0                                           // 0000000044E8: BFC60000
	s_wait_alu 0xfffe                                          // 0000000044EC: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v12, s12                          // 0000000044F0: D7006A05 0000190C
	s_wait_alu 0xfffd                                          // 0000000044F8: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s13, v13, vcc_lo             // 0000000044FC: D5207C06 01AA1A0D
	global_load_b32 v5, v[5:6], off                            // 000000004504: EE05007C 00000005 00000005
	v_mov_b32_e32 v6, s9                                       // 000000004510: 7E0C0209
	s_add_co_i32 s9, s9, 4                                     // 000000004514: 81098409
	s_cmp_lg_u32 s8, 0                                         // 000000004518: BF078008
	ds_load_b32 v6, v6                                         // 00000000451C: D8D80000 06000006
	s_wait_loadcnt_dscnt 0x0                                   // 000000004524: BFC80000
	v_fmac_f32_e32 v4, v6, v5                                  // 000000004528: 56080B06
	s_cbranch_scc1 65507                                       // 00000000452C: BFA2FFE3 <ullm_sq8_0_flash2_legacy_reference_kernel+0xebc>
	s_wait_dscnt 0x0                                           // 000000004530: BFC60000
	ds_load_b32 v5, v11 offset:1280                            // 000000004534: D8D80500 0500000B
	s_wait_dscnt 0x0                                           // 00000000453C: BFC60000
	v_fmac_f32_e32 v4, v9, v5                                  // 000000004540: 56080B09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004544: BF870001
	v_mov_b32_e32 v9, v4                                       // 000000004548: 7E120304
	s_wait_alu 0xfffe                                          // 00000000454C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s6                              // 000000004550: 8C7E067E
	ds_load_b32 v4, v11 offset:1292                            // 000000004554: D8D8050C 0400000B
	s_add_nc_u64 s[28:29], s[28:29], 64                        // 00000000455C: A99CC01C
	s_wait_loadcnt_dscnt 0x0                                   // 000000004560: BFC80000
	s_wait_alu 0xfffe                                          // 000000004564: BF88FFFE
	v_cmp_ge_u64_e64 s6, s[28:29], s[14:15]                    // 000000004568: D45E0006 00001C1C
	s_barrier_signal -1                                        // 000000004570: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004574: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004578: EE0AC07C 00040000 00000000
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000004584: 8B6A067E
	v_fmac_f32_e32 v4, v15, v5                                 // 000000004588: 56080B0F
	s_wait_alu 0xfffe                                          // 00000000458C: BF88FFFE
	s_cbranch_vccnz 3                                          // 000000004590: BFA40003 <ullm_sq8_0_flash2_legacy_reference_kernel+0xfa0>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004594: BF870001
	v_mov_b32_e32 v15, v4                                      // 000000004598: 7E1E0304
	s_branch 64973                                             // 00000000459C: BFA0FDCD <ullm_sq8_0_flash2_legacy_reference_kernel+0x6d4>
	s_and_saveexec_b32 s3, s2                                  // 0000000045A0: BE832002
	s_cbranch_execz 34                                         // 0000000045A4: BFA50022 <ullm_sq8_0_flash2_legacy_reference_kernel+0x1030>
	v_div_scale_f32 v0, null, v4, v4, v9                       // 0000000045A8: D6FC7C00 04260904
	s_load_b64 s[0:1], s[0:1], 0x50                            // 0000000045B0: F4002000 F8000050
	s_mul_u64 s[2:3], s[16:17], s[24:25]                       // 0000000045B8: AA821810
	s_wait_alu 0xfffe                                          // 0000000045BC: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000045C0: 84828202
	v_rcp_f32_e32 v1, v0                                       // 0000000045C4: 7E025500
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000045C8: BF870095
	v_fma_f32 v2, -v0, v1, 1.0                                 // 0000000045CC: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 0000000045D4: 56020302
	v_div_scale_f32 v2, vcc_lo, v9, v4, v9                     // 0000000045D8: D6FC6A02 04260909
	s_wait_kmcnt 0x0                                           // 0000000045E0: BFC70000
	s_wait_alu 0xfffe                                          // 0000000045E4: BF88FFFE
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 0000000045E8: A9800200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000045EC: BF870091
	v_mul_f32_e32 v3, v2, v1                                   // 0000000045F0: 10060302
	v_fma_f32 v5, -v0, v3, v2                                  // 0000000045F4: D6130005 240A0700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000045FC: BF870091
	v_fmac_f32_e32 v3, v5, v1                                  // 000000004600: 56060305
	v_fma_f32 v0, -v0, v3, v2                                  // 000000004604: D6130000 240A0700
	s_wait_alu 0xfffd                                          // 00000000460C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004610: BF870091
	v_div_fmas_f32 v0, v0, v1, v3                              // 000000004614: D6370000 040E0300
	v_div_fixup_f32 v0, v0, v4, v9                             // 00000000461C: D6270000 04260900
	global_store_b32 v8, v0, s[0:1]                            // 000000004624: EE068000 00000000 00000008
	s_endpgm                                                   // 000000004630: BFB00000
	s_branch 64691                                             // 000000004634: BFA0FCB3 <ullm_sq8_0_flash2_legacy_reference_kernel+0x304>
	s_branch 64859                                             // 000000004638: BFA0FD5B <ullm_sq8_0_flash2_legacy_reference_kernel+0x5a8>
	s_nop 0                                                    // 00000000463C: BF800000
	s_nop 0                                                    // 000000004640: BF800000
	s_nop 0                                                    // 000000004644: BF800000
	s_nop 0                                                    // 000000004648: BF800000
	s_nop 0                                                    // 00000000464C: BF800000
	s_nop 0                                                    // 000000004650: BF800000
	s_nop 0                                                    // 000000004654: BF800000
	s_nop 0                                                    // 000000004658: BF800000
	s_nop 0                                                    // 00000000465C: BF800000
	s_nop 0                                                    // 000000004660: BF800000
	s_nop 0                                                    // 000000004664: BF800000
	s_nop 0                                                    // 000000004668: BF800000
	s_nop 0                                                    // 00000000466C: BF800000
	s_nop 0                                                    // 000000004670: BF800000
	s_nop 0                                                    // 000000004674: BF800000
	s_nop 0                                                    // 000000004678: BF800000
	s_nop 0                                                    // 00000000467C: BF800000
	s_nop 0                                                    // 000000004680: BF800000
	s_nop 0                                                    // 000000004684: BF800000
	s_nop 0                                                    // 000000004688: BF800000
	s_nop 0                                                    // 00000000468C: BF800000
	s_nop 0                                                    // 000000004690: BF800000
	s_nop 0                                                    // 000000004694: BF800000
	s_nop 0                                                    // 000000004698: BF800000
	s_nop 0                                                    // 00000000469C: BF800000
	s_nop 0                                                    // 0000000046A0: BF800000
	s_nop 0                                                    // 0000000046A4: BF800000
	s_nop 0                                                    // 0000000046A8: BF800000
	s_nop 0                                                    // 0000000046AC: BF800000
	s_nop 0                                                    // 0000000046B0: BF800000
	s_nop 0                                                    // 0000000046B4: BF800000
	s_nop 0                                                    // 0000000046B8: BF800000
	s_nop 0                                                    // 0000000046BC: BF800000
	s_nop 0                                                    // 0000000046C0: BF800000
	s_nop 0                                                    // 0000000046C4: BF800000
	s_nop 0                                                    // 0000000046C8: BF800000
	s_nop 0                                                    // 0000000046CC: BF800000
	s_nop 0                                                    // 0000000046D0: BF800000
	s_nop 0                                                    // 0000000046D4: BF800000
	s_nop 0                                                    // 0000000046D8: BF800000
	s_nop 0                                                    // 0000000046DC: BF800000
	s_nop 0                                                    // 0000000046E0: BF800000
	s_nop 0                                                    // 0000000046E4: BF800000
	s_nop 0                                                    // 0000000046E8: BF800000
	s_nop 0                                                    // 0000000046EC: BF800000
	s_nop 0                                                    // 0000000046F0: BF800000
	s_nop 0                                                    // 0000000046F4: BF800000
	s_nop 0                                                    // 0000000046F8: BF800000
	s_nop 0                                                    // 0000000046FC: BF800000

0000000000004700 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel>:
	s_load_b512 s[12:27], s[0:1], 0x0                          // 000000004700: F4008300 F8000000
	s_mov_b32 s31, 0                                           // 000000004708: BE9F0080
	s_mov_b32 s28, ttmp9                                       // 00000000470C: BE9C0075
	s_mov_b32 s29, s31                                         // 000000004710: BE9D001F
	s_wait_kmcnt 0x0                                           // 000000004714: BFC70000
	s_mul_u64 s[2:3], s[22:23], s[20:21]                       // 000000004718: AA821416
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000471C: BF870009
	v_cmp_le_u64_e64 s2, s[2:3], s[28:29]                      // 000000004720: D45B0002 00003802
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000004728: 8B6A027E
	s_cbranch_vccnz 1098                                       // 00000000472C: BFA4044A <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1158>
	s_clause 0x1                                               // 000000004730: BF850001
	s_load_b32 s2, s[0:1], 0x64                                // 000000004734: F4000080 F8000064
	s_load_b64 s[20:21], s[0:1], 0x40                          // 00000000473C: F4002500 F8000040
	s_wait_kmcnt 0x0                                           // 000000004744: BFC70000
	s_and_b32 s30, s2, 0xffff                                  // 000000004748: 8B1EFF02 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004750: BF870009
	v_cmp_gt_u64_e64 s2, s[20:21], s[30:31]                    // 000000004754: D45C0002 00003C14
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000475C: 8B6A027E
	s_cbranch_vccnz 1085                                       // 000000004760: BFA4043D <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1158>
	v_cmp_lt_u64_e64 s2, s[28:29], s[22:23]                    // 000000004764: D4590002 00002C1C
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000476C: 8B6A027E
	s_mov_b64 s[2:3], 0                                        // 000000004770: BE820180
	s_cbranch_vccnz 32                                         // 000000004774: BFA40020 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xf8>
	v_cvt_f32_u32_e32 v1, s22                                  // 000000004778: 7E020C16
	s_sub_co_i32 s3, 0, s22                                    // 00000000477C: 81831680
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004780: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000004784: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000004788: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004790: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000004794: 7E020F01
	v_readfirstlane_b32 s2, v1                                 // 000000004798: 7E040501
	s_wait_alu 0xfffe                                          // 00000000479C: BF88FFFE
	s_mul_i32 s3, s3, s2                                       // 0000000047A0: 96030203
	s_wait_alu 0xfffe                                          // 0000000047A4: BF88FFFE
	s_mul_hi_u32 s3, s2, s3                                    // 0000000047A8: 96830302
	s_wait_alu 0xfffe                                          // 0000000047AC: BF88FFFE
	s_add_co_i32 s2, s2, s3                                    // 0000000047B0: 81020302
	s_wait_alu 0xfffe                                          // 0000000047B4: BF88FFFE
	s_mul_hi_u32 s2, s28, s2                                   // 0000000047B8: 9682021C
	s_wait_alu 0xfffe                                          // 0000000047BC: BF88FFFE
	s_mul_i32 s3, s2, s22                                      // 0000000047C0: 96031602
	s_add_co_i32 s4, s2, 1                                     // 0000000047C4: 81048102
	s_wait_alu 0xfffe                                          // 0000000047C8: BF88FFFE
	s_sub_co_i32 s3, s28, s3                                   // 0000000047CC: 8183031C
	s_wait_alu 0xfffe                                          // 0000000047D0: BF88FFFE
	s_sub_co_i32 s5, s3, s22                                   // 0000000047D4: 81851603
	s_cmp_ge_u32 s3, s22                                       // 0000000047D8: BF091603
	s_cselect_b32 s2, s4, s2                                   // 0000000047DC: 98020204
	s_cselect_b32 s3, s5, s3                                   // 0000000047E0: 98030305
	s_wait_alu 0xfffe                                          // 0000000047E4: BF88FFFE
	s_add_co_i32 s4, s2, 1                                     // 0000000047E8: 81048102
	s_cmp_ge_u32 s3, s22                                       // 0000000047EC: BF091603
	s_mov_b32 s3, 0                                            // 0000000047F0: BE830080
	s_cselect_b32 s2, s4, s2                                   // 0000000047F4: 98020204
	s_or_b64 s[6:7], s[22:23], s[24:25]                        // 0000000047F8: 8C861816
	s_mov_b32 s6, 0                                            // 0000000047FC: BE860080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004800: BF870009
	s_cmp_lg_u64 s[6:7], 0                                     // 000000004804: BF118006
	s_cbranch_scc0 1044                                        // 000000004808: BFA10414 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x115c>
	s_cvt_f32_u32 s4, s24                                      // 00000000480C: BE846518
	s_cvt_f32_u32 s5, s25                                      // 000000004810: BE856519
	s_sub_nc_u64 s[8:9], 0, s[24:25]                           // 000000004814: AA081880
	s_mov_b32 s11, s6                                          // 000000004818: BE8B0006
	s_mov_b32 s37, s6                                          // 00000000481C: BEA50006
	s_fmamk_f32 s4, s5, 0x4f800000, s4                         // 000000004820: A3040405 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004828: BF87029B
	v_s_rcp_f32 s4, s4                                         // 00000000482C: D6840004 00000004
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000004834: A204FF04 5F7FFFFC
	s_wait_alu 0xfffe                                          // 00000000483C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000004840: BF87052A
	s_mul_f32 s5, s4, 0x2f800000                               // 000000004844: A205FF04 2F800000
	s_wait_alu 0xfffe                                          // 00000000484C: BF88FFFE
	s_trunc_f32 s5, s5                                         // 000000004850: BE856205
	s_wait_alu 0xfffe                                          // 000000004854: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000004858: BF8704BA
	s_fmamk_f32 s4, s5, 0xcf800000, s4                         // 00000000485C: A3040405 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000004864: BE856705
	s_wait_alu 0xfffe                                          // 000000004868: BF88FFFE
	s_cvt_u32_f32 s4, s4                                       // 00000000486C: BE846704
	s_wait_alu 0xfffe                                          // 000000004870: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000004874: BF87049A
	s_mul_u64 s[34:35], s[8:9], s[4:5]                         // 000000004878: AAA20408
	s_mul_hi_u32 s39, s4, s35                                  // 00000000487C: 96A72304
	s_mul_i32 s38, s4, s35                                     // 000000004880: 96262304
	s_mul_hi_u32 s10, s4, s34                                  // 000000004884: 968A2204
	s_mul_i32 s31, s5, s34                                     // 000000004888: 961F2205
	s_add_nc_u64 s[10:11], s[10:11], s[38:39]                  // 00000000488C: A98A260A
	s_mul_hi_u32 s7, s5, s34                                   // 000000004890: 96872205
	s_mul_hi_u32 s33, s5, s35                                  // 000000004894: 96A12305
	s_wait_alu 0xfffe                                          // 000000004898: BF88FFFE
	s_add_co_u32 s10, s10, s31                                 // 00000000489C: 800A1F0A
	s_add_co_ci_u32 s36, s11, s7                               // 0000000048A0: 8224070B
	s_mul_i32 s34, s5, s35                                     // 0000000048A4: 96222305
	s_add_co_ci_u32 s35, s33, 0                                // 0000000048A8: 82238021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000048AC: BF8704C9
	s_add_nc_u64 s[10:11], s[36:37], s[34:35]                  // 0000000048B0: A98A2224
	s_mov_b32 s35, s6                                          // 0000000048B4: BEA30006
	s_add_co_u32 s4, s4, s10                                   // 0000000048B8: 80040A04
	s_cselect_b32 s7, -1, 0                                    // 0000000048BC: 980780C1
	s_cmp_lg_u32 s7, 0                                         // 0000000048C0: BF078007
	s_add_co_ci_u32 s5, s5, s11                                // 0000000048C4: 82050B05
	s_mov_b32 s11, s6                                          // 0000000048C8: BE8B0006
	s_wait_alu 0xfffe                                          // 0000000048CC: BF88FFFE
	s_mul_u64 s[8:9], s[8:9], s[4:5]                           // 0000000048D0: AA880408
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000048D4: BF870009
	s_mul_hi_u32 s37, s4, s9                                   // 0000000048D8: 96A50904
	s_mul_i32 s36, s4, s9                                      // 0000000048DC: 96240904
	s_mul_hi_u32 s10, s4, s8                                   // 0000000048E0: 968A0804
	s_mul_i32 s31, s5, s8                                      // 0000000048E4: 961F0805
	s_add_nc_u64 s[10:11], s[10:11], s[36:37]                  // 0000000048E8: A98A240A
	s_mul_hi_u32 s7, s5, s8                                    // 0000000048EC: 96870805
	s_mul_hi_u32 s33, s5, s9                                   // 0000000048F0: 96A10905
	s_mul_i32 s8, s5, s9                                       // 0000000048F4: 96080905
	s_wait_alu 0xfffe                                          // 0000000048F8: BF88FFFE
	s_add_co_u32 s9, s10, s31                                  // 0000000048FC: 80091F0A
	s_add_co_ci_u32 s34, s11, s7                               // 000000004900: 8222070B
	s_add_co_ci_u32 s9, s33, 0                                 // 000000004904: 82098021
	s_mov_b32 s11, s6                                          // 000000004908: BE8B0006
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]                      // 00000000490C: A9880822
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004910: BF870009
	s_add_co_u32 s4, s4, s8                                    // 000000004914: 80040804
	s_cselect_b32 s7, -1, 0                                    // 000000004918: 980780C1
	s_wait_alu 0xfffe                                          // 00000000491C: BF88FFFE
	s_mul_hi_u32 s10, s22, s4                                  // 000000004920: 968A0416
	s_cmp_lg_u32 s7, 0                                         // 000000004924: BF078007
	s_mul_hi_u32 s7, s23, s4                                   // 000000004928: 96870417
	s_add_co_ci_u32 s8, s5, s9                                 // 00000000492C: 82080905
	s_mul_i32 s9, s23, s4                                      // 000000004930: 96090417
	s_mul_hi_u32 s5, s22, s8                                   // 000000004934: 96850816
	s_mul_i32 s4, s22, s8                                      // 000000004938: 96040816
	s_mul_hi_u32 s31, s23, s8                                  // 00000000493C: 969F0817
	s_wait_alu 0xfffe                                          // 000000004940: BF88FFFE
	s_add_nc_u64 s[4:5], s[10:11], s[4:5]                      // 000000004944: A984040A
	s_mul_i32 s8, s23, s8                                      // 000000004948: 96080817
	s_wait_alu 0xfffe                                          // 00000000494C: BF88FFFE
	s_add_co_u32 s4, s4, s9                                    // 000000004950: 80040904
	s_add_co_ci_u32 s34, s5, s7                                // 000000004954: 82220705
	s_add_co_ci_u32 s9, s31, 0                                 // 000000004958: 8209801F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000495C: BF870009
	s_add_nc_u64 s[4:5], s[34:35], s[8:9]                      // 000000004960: A9840822
	s_wait_alu 0xfffe                                          // 000000004964: BF88FFFE
	s_mul_u64 s[8:9], s[24:25], s[4:5]                         // 000000004968: AA880418
	s_add_nc_u64 s[34:35], s[4:5], 2                           // 00000000496C: A9A28204
	s_sub_co_u32 s7, s22, s8                                   // 000000004970: 80870816
	s_cselect_b32 s8, -1, 0                                    // 000000004974: 980880C1
	s_sub_co_i32 s10, s23, s9                                  // 000000004978: 818A0917
	s_cmp_lg_u32 s8, 0                                         // 00000000497C: BF078008
	s_sub_co_ci_u32 s10, s10, s25                              // 000000004980: 828A190A
	s_sub_co_u32 s11, s7, s24                                  // 000000004984: 808B1807
	s_cselect_b32 s31, -1, 0                                   // 000000004988: 981F80C1
	s_wait_alu 0xfffe                                          // 00000000498C: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000004990: BF07801F
	s_sub_co_ci_u32 s10, s10, 0                                // 000000004994: 828A800A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004998: BF870009
	s_cmp_ge_u32 s10, s25                                      // 00000000499C: BF09190A
	s_cselect_b32 s31, -1, 0                                   // 0000000049A0: 981F80C1
	s_cmp_ge_u32 s11, s24                                      // 0000000049A4: BF09180B
	s_cselect_b32 s33, -1, 0                                   // 0000000049A8: 982180C1
	s_cmp_eq_u32 s10, s25                                      // 0000000049AC: BF06190A
	s_add_nc_u64 s[10:11], s[4:5], 1                           // 0000000049B0: A98A8104
	s_wait_alu 0xfffe                                          // 0000000049B4: BF88FFFE
	s_cselect_b32 s31, s33, s31                                // 0000000049B8: 981F1F21
	s_wait_alu 0xfffe                                          // 0000000049BC: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 0000000049C0: BF07801F
	s_cselect_b32 s10, s34, s10                                // 0000000049C4: 980A0A22
	s_cselect_b32 s11, s35, s11                                // 0000000049C8: 980B0B23
	s_cmp_lg_u32 s8, 0                                         // 0000000049CC: BF078008
	s_sub_co_ci_u32 s8, s23, s9                                // 0000000049D0: 82880917
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000049D4: BF870009
	s_cmp_ge_u32 s8, s25                                       // 0000000049D8: BF091908
	s_cselect_b32 s9, -1, 0                                    // 0000000049DC: 980980C1
	s_cmp_ge_u32 s7, s24                                       // 0000000049E0: BF091807
	s_cselect_b32 s7, -1, 0                                    // 0000000049E4: 980780C1
	s_cmp_eq_u32 s8, s25                                       // 0000000049E8: BF061908
	s_cselect_b32 s7, s7, s9                                   // 0000000049EC: 98070907
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000049F0: BF870009
	s_cmp_lg_u32 s7, 0                                         // 0000000049F4: BF078007
	s_cselect_b32 s5, s11, s5                                  // 0000000049F8: 9805050B
	s_cselect_b32 s4, s10, s4                                  // 0000000049FC: 9804040A
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000004A00: 916A067E
	s_cbranch_vccnz 32                                         // 000000004A04: BFA40020 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x388>
	v_cvt_f32_u32_e32 v1, s24                                  // 000000004A08: 7E020C18
	s_sub_co_i32 s5, 0, s24                                    // 000000004A0C: 81851880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004A10: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000004A14: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000004A18: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004A20: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000004A24: 7E020F01
	v_readfirstlane_b32 s4, v1                                 // 000000004A28: 7E080501
	s_wait_alu 0xfffe                                          // 000000004A2C: BF88FFFE
	s_mul_i32 s5, s5, s4                                       // 000000004A30: 96050405
	s_wait_alu 0xfffe                                          // 000000004A34: BF88FFFE
	s_mul_hi_u32 s5, s4, s5                                    // 000000004A38: 96850504
	s_wait_alu 0xfffe                                          // 000000004A3C: BF88FFFE
	s_add_co_i32 s4, s4, s5                                    // 000000004A40: 81040504
	s_wait_alu 0xfffe                                          // 000000004A44: BF88FFFE
	s_mul_hi_u32 s4, s22, s4                                   // 000000004A48: 96840416
	s_wait_alu 0xfffe                                          // 000000004A4C: BF88FFFE
	s_mul_i32 s5, s4, s24                                      // 000000004A50: 96051804
	s_add_co_i32 s6, s4, 1                                     // 000000004A54: 81068104
	s_wait_alu 0xfffe                                          // 000000004A58: BF88FFFE
	s_sub_co_i32 s5, s22, s5                                   // 000000004A5C: 81850516
	s_wait_alu 0xfffe                                          // 000000004A60: BF88FFFE
	s_sub_co_i32 s7, s5, s24                                   // 000000004A64: 81871805
	s_cmp_ge_u32 s5, s24                                       // 000000004A68: BF091805
	s_cselect_b32 s4, s6, s4                                   // 000000004A6C: 98040406
	s_cselect_b32 s5, s7, s5                                   // 000000004A70: 98050507
	s_wait_alu 0xfffe                                          // 000000004A74: BF88FFFE
	s_add_co_i32 s6, s4, 1                                     // 000000004A78: 81068104
	s_cmp_ge_u32 s5, s24                                       // 000000004A7C: BF091805
	s_mov_b32 s5, 0                                            // 000000004A80: BE850080
	s_cselect_b32 s4, s6, s4                                   // 000000004A84: 98040406
	s_mul_u64 s[6:7], s[2:3], s[22:23]                         // 000000004A88: AA861602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000004A8C: BF8704C9
	s_sub_nc_u64 s[6:7], s[28:29], s[6:7]                      // 000000004A90: AA06061C
	s_wait_alu 0xfffe                                          // 000000004A94: BF88FFFE
	s_or_b64 s[8:9], s[6:7], s[4:5]                            // 000000004A98: 8C880406
	s_mov_b32 s8, 0                                            // 000000004A9C: BE880080
	s_cmp_lg_u64 s[8:9], 0                                     // 000000004AA0: BF118008
	s_cbranch_scc0 878                                         // 000000004AA4: BFA1036E <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1160>
	s_cvt_f32_u32 s9, s4                                       // 000000004AA8: BE896504
	s_cvt_f32_u32 s10, s5                                      // 000000004AAC: BE8A6505
	s_sub_nc_u64 s[22:23], 0, s[4:5]                           // 000000004AB0: AA160480
	s_mov_b32 s35, s8                                          // 000000004AB4: BEA30008
	s_mov_b32 s39, s8                                          // 000000004AB8: BEA70008
	s_fmamk_f32 s9, s10, 0x4f800000, s9                        // 000000004ABC: A309090A 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004AC4: BF87029B
	v_s_rcp_f32 s9, s9                                         // 000000004AC8: D6840009 00000009
	s_mul_f32 s9, s9, 0x5f7ffffc                               // 000000004AD0: A209FF09 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000004AD8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000004ADC: BF87059A
	s_mul_f32 s10, s9, 0x2f800000                              // 000000004AE0: A20AFF09 2F800000
	s_trunc_f32 s10, s10                                       // 000000004AE8: BE8A620A
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000004AEC: BF8704BB
	s_fmamk_f32 s9, s10, 0xcf800000, s9                        // 000000004AF0: A309090A CF800000
	s_cvt_u32_f32 s11, s10                                     // 000000004AF8: BE8B670A
	s_wait_alu 0xfffe                                          // 000000004AFC: BF88FFFE
	s_cvt_u32_f32 s10, s9                                      // 000000004B00: BE8A6709
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000004B04: BF87049B
	s_mul_u64 s[36:37], s[22:23], s[10:11]                     // 000000004B08: AAA40A16
	s_mul_hi_u32 s41, s10, s37                                 // 000000004B0C: 96A9250A
	s_mul_i32 s40, s10, s37                                    // 000000004B10: 9628250A
	s_mul_hi_u32 s34, s10, s36                                 // 000000004B14: 96A2240A
	s_mul_i32 s31, s11, s36                                    // 000000004B18: 961F240B
	s_add_nc_u64 s[34:35], s[34:35], s[40:41]                  // 000000004B1C: A9A22822
	s_mul_hi_u32 s9, s11, s36                                  // 000000004B20: 9689240B
	s_mul_hi_u32 s33, s11, s37                                 // 000000004B24: 96A1250B
	s_wait_alu 0xfffe                                          // 000000004B28: BF88FFFE
	s_add_co_u32 s31, s34, s31                                 // 000000004B2C: 801F1F22
	s_add_co_ci_u32 s38, s35, s9                               // 000000004B30: 82260923
	s_mul_i32 s36, s11, s37                                    // 000000004B34: 9624250B
	s_add_co_ci_u32 s37, s33, 0                                // 000000004B38: 82258021
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004B3C: BF870009
	s_add_nc_u64 s[34:35], s[38:39], s[36:37]                  // 000000004B40: A9A22426
	s_mov_b32 s37, s8                                          // 000000004B44: BEA50008
	s_add_co_u32 s10, s10, s34                                 // 000000004B48: 800A220A
	s_cselect_b32 s9, -1, 0                                    // 000000004B4C: 980980C1
	s_wait_alu 0xfffe                                          // 000000004B50: BF88FFFE
	s_cmp_lg_u32 s9, 0                                         // 000000004B54: BF078009
	s_add_co_ci_u32 s11, s11, s35                              // 000000004B58: 820B230B
	s_mov_b32 s35, s8                                          // 000000004B5C: BEA30008
	s_mul_u64 s[22:23], s[22:23], s[10:11]                     // 000000004B60: AA960A16
	s_wait_alu 0xfffe                                          // 000000004B64: BF88FFFE
	s_mul_hi_u32 s39, s10, s23                                 // 000000004B68: 96A7170A
	s_mul_i32 s38, s10, s23                                    // 000000004B6C: 9626170A
	s_mul_hi_u32 s34, s10, s22                                 // 000000004B70: 96A2160A
	s_mul_i32 s31, s11, s22                                    // 000000004B74: 961F160B
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]                  // 000000004B78: A9A22622
	s_mul_hi_u32 s9, s11, s22                                  // 000000004B7C: 9689160B
	s_mul_hi_u32 s33, s11, s23                                 // 000000004B80: 96A1170B
	s_mul_i32 s22, s11, s23                                    // 000000004B84: 9616170B
	s_wait_alu 0xfffe                                          // 000000004B88: BF88FFFE
	s_add_co_u32 s23, s34, s31                                 // 000000004B8C: 80171F22
	s_add_co_ci_u32 s36, s35, s9                               // 000000004B90: 82240923
	s_add_co_ci_u32 s23, s33, 0                                // 000000004B94: 82178021
	s_mov_b32 s35, s8                                          // 000000004B98: BEA30008
	s_wait_alu 0xfffe                                          // 000000004B9C: BF88FFFE
	s_add_nc_u64 s[22:23], s[36:37], s[22:23]                  // 000000004BA0: A9961624
	s_wait_alu 0xfffe                                          // 000000004BA4: BF88FFFE
	s_add_co_u32 s9, s10, s22                                  // 000000004BA8: 8009160A
	s_cselect_b32 s10, -1, 0                                   // 000000004BAC: 980A80C1
	s_wait_alu 0xfffe                                          // 000000004BB0: BF88FFFE
	s_mul_hi_u32 s34, s6, s9                                   // 000000004BB4: 96A20906
	s_cmp_lg_u32 s10, 0                                        // 000000004BB8: BF07800A
	s_mul_hi_u32 s31, s7, s9                                   // 000000004BBC: 969F0907
	s_add_co_ci_u32 s22, s11, s23                              // 000000004BC0: 8216170B
	s_mul_i32 s9, s7, s9                                       // 000000004BC4: 96090907
	s_wait_alu 0xfffe                                          // 000000004BC8: BF88FFFE
	s_mul_hi_u32 s11, s6, s22                                  // 000000004BCC: 968B1606
	s_mul_i32 s10, s6, s22                                     // 000000004BD0: 960A1606
	s_mul_hi_u32 s23, s7, s22                                  // 000000004BD4: 96971607
	s_add_nc_u64 s[10:11], s[34:35], s[10:11]                  // 000000004BD8: A98A0A22
	s_mul_i32 s22, s7, s22                                     // 000000004BDC: 96161607
	s_add_co_u32 s9, s10, s9                                   // 000000004BE0: 8009090A
	s_add_co_ci_u32 s36, s11, s31                              // 000000004BE4: 82241F0B
	s_wait_alu 0xfffe                                          // 000000004BE8: BF88FFFE
	s_add_co_ci_u32 s23, s23, 0                                // 000000004BEC: 82178017
	s_wait_alu 0xfffe                                          // 000000004BF0: BF88FFFE
	s_add_nc_u64 s[10:11], s[36:37], s[22:23]                  // 000000004BF4: A98A1624
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004BF8: BF870009
	s_mul_u64 s[22:23], s[4:5], s[10:11]                       // 000000004BFC: AA960A04
	s_wait_alu 0xfffe                                          // 000000004C00: BF88FFFE
	s_sub_co_u32 s9, s6, s22                                   // 000000004C04: 80891606
	s_cselect_b32 s22, -1, 0                                   // 000000004C08: 981680C1
	s_sub_co_i32 s31, s7, s23                                  // 000000004C0C: 819F1707
	s_wait_alu 0xfffe                                          // 000000004C10: BF88FFFE
	s_cmp_lg_u32 s22, 0                                        // 000000004C14: BF078016
	s_sub_co_ci_u32 s31, s31, s5                               // 000000004C18: 829F051F
	s_sub_co_u32 s33, s9, s4                                   // 000000004C1C: 80A10409
	s_cselect_b32 s34, -1, 0                                   // 000000004C20: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004C24: BF870009
	s_cmp_lg_u32 s34, 0                                        // 000000004C28: BF078022
	s_add_nc_u64 s[34:35], s[10:11], 1                         // 000000004C2C: A9A2810A
	s_wait_alu 0xfffe                                          // 000000004C30: BF88FFFE
	s_sub_co_ci_u32 s31, s31, 0                                // 000000004C34: 829F801F
	s_wait_alu 0xfffe                                          // 000000004C38: BF88FFFE
	s_cmp_ge_u32 s31, s5                                       // 000000004C3C: BF09051F
	s_cselect_b32 s36, -1, 0                                   // 000000004C40: 982480C1
	s_cmp_ge_u32 s33, s4                                       // 000000004C44: BF090421
	s_cselect_b32 s33, -1, 0                                   // 000000004C48: 982180C1
	s_cmp_eq_u32 s31, s5                                       // 000000004C4C: BF06051F
	s_cselect_b32 s31, s33, s36                                // 000000004C50: 981F2421
	s_add_nc_u64 s[36:37], s[10:11], 2                         // 000000004C54: A9A4820A
	s_wait_alu 0xfffe                                          // 000000004C58: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000004C5C: BF07801F
	s_cselect_b32 s31, s36, s34                                // 000000004C60: 981F2224
	s_cselect_b32 s33, s37, s35                                // 000000004C64: 98212325
	s_cmp_lg_u32 s22, 0                                        // 000000004C68: BF078016
	s_sub_co_ci_u32 s7, s7, s23                                // 000000004C6C: 82871707
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004C70: BF870009
	s_cmp_ge_u32 s7, s5                                        // 000000004C74: BF090507
	s_cselect_b32 s22, -1, 0                                   // 000000004C78: 981680C1
	s_cmp_ge_u32 s9, s4                                        // 000000004C7C: BF090409
	s_cselect_b32 s9, -1, 0                                    // 000000004C80: 980980C1
	s_cmp_eq_u32 s7, s5                                        // 000000004C84: BF060507
	s_wait_alu 0xfffe                                          // 000000004C88: BF88FFFE
	s_cselect_b32 s5, s9, s22                                  // 000000004C8C: 98051609
	s_wait_alu 0xfffe                                          // 000000004C90: BF88FFFE
	s_cmp_lg_u32 s5, 0                                         // 000000004C94: BF078005
	s_cselect_b32 s23, s33, s11                                // 000000004C98: 98170B21
	s_cselect_b32 s22, s31, s10                                // 000000004C9C: 98160A1F
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000004CA0: 916A087E
	s_cbranch_vccnz 29                                         // 000000004CA4: BFA4001D <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x61c>
	v_cvt_f32_u32_e32 v1, s4                                   // 000000004CA8: 7E020C04
	s_sub_co_i32 s7, 0, s4                                     // 000000004CAC: 81870480
	s_mov_b32 s23, 0                                           // 000000004CB0: BE970080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004CB4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000004CB8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000004CBC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004CC4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000004CC8: 7E020F01
	v_readfirstlane_b32 s5, v1                                 // 000000004CCC: 7E0A0501
	s_mul_i32 s7, s7, s5                                       // 000000004CD0: 96070507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000004CD4: BF870499
	s_mul_hi_u32 s7, s5, s7                                    // 000000004CD8: 96870705
	s_add_co_i32 s5, s5, s7                                    // 000000004CDC: 81050705
	s_wait_alu 0xfffe                                          // 000000004CE0: BF88FFFE
	s_mul_hi_u32 s5, s6, s5                                    // 000000004CE4: 96850506
	s_wait_alu 0xfffe                                          // 000000004CE8: BF88FFFE
	s_mul_i32 s7, s5, s4                                       // 000000004CEC: 96070405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004CF0: BF870009
	s_sub_co_i32 s6, s6, s7                                    // 000000004CF4: 81860706
	s_add_co_i32 s7, s5, 1                                     // 000000004CF8: 81078105
	s_sub_co_i32 s8, s6, s4                                    // 000000004CFC: 81880406
	s_cmp_ge_u32 s6, s4                                        // 000000004D00: BF090406
	s_cselect_b32 s5, s7, s5                                   // 000000004D04: 98050507
	s_wait_alu 0xfffe                                          // 000000004D08: BF88FFFE
	s_cselect_b32 s6, s8, s6                                   // 000000004D0C: 98060608
	s_add_co_i32 s7, s5, 1                                     // 000000004D10: 81078105
	s_cmp_ge_u32 s6, s4                                        // 000000004D14: BF090406
	s_cselect_b32 s22, s7, s5                                  // 000000004D18: 98160507
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v8, 2, v0       // 000000004D1C: CA220080 01080082
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]                      // 000000004D24: A9820212
	s_mov_b64 s[34:35], 0                                      // 000000004D28: BEA20180
	s_wait_alu 0xfffe                                          // 000000004D2C: BF88FFFE
	s_add_nc_u64 s[18:19], s[2:3], 1                           // 000000004D30: A9928102
	v_cmp_gt_u64_e64 s2, s[20:21], v[0:1]                      // 000000004D34: D45C0002 00020014
	v_cmp_le_u64_e64 s3, s[20:21], v[0:1]                      // 000000004D3C: D45B0003 00020014
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v13, v1            // 000000004D44: CA100101 040C0101
	s_cmp_eq_u64 s[18:19], 0                                   // 000000004D4C: BF108012
	s_cbranch_scc1 669                                         // 000000004D50: BFA2029D <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x10c8>
	v_dual_mov_b32 v13, 0 :: v_dual_and_b32 v2, 31, v0         // 000000004D54: CA240080 0D02009F
	v_mbcnt_lo_u32_b32 v3, -1, 0                               // 000000004D5C: D71F0003 000100C1
	s_load_b32 s31, s[0:1], 0x48                               // 000000004D64: F40007C0 F8000048
	v_add_co_u32 v14, s9, s16, v8                              // 000000004D6C: D700090E 00021010
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004D74: BF870003
	v_cmp_eq_u32_e64 s5, 0, v2                                 // 000000004D78: D44A0005 00020480
	v_cmp_gt_u32_e64 s7, 8, v2                                 // 000000004D80: D44C0007 00020488
	v_lshlrev_b32_e32 v10, 2, v2                               // 000000004D88: 30140482
	v_xor_b32_e32 v2, 31, v3                                   // 000000004D8C: 3A04069F
	s_cmp_gt_u32 s30, 1                                        // 000000004D90: BF08811E
	v_add_co_ci_u32_e64 v15, null, s17, 0, s9                  // 000000004D94: D5207C0F 00250011
	s_mul_u64 s[16:17], s[26:27], s[28:29]                     // 000000004D9C: AA901C1A
	v_cmp_gt_u32_e32 vcc_lo, 8, v2                             // 000000004DA0: 7C980488
	v_and_b32_e32 v4, 16, v2                                   // 000000004DA4: 36080490
	s_cselect_b32 s33, -1, 0                                   // 000000004DA8: 982180C1
	s_wait_alu 0xfffe                                          // 000000004DAC: BF88FFFE
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000004DB0: 84908210
	v_dual_mov_b32 v12, 0 :: v_dual_lshlrev_b32 v11, 2, v0     // 000000004DB4: CA220080 0C0A0082
	v_cndmask_b32_e64 v5, 8, 0, vcc_lo                         // 000000004DBC: D5010005 01A90088
	v_cmp_gt_u32_e32 vcc_lo, 4, v2                             // 000000004DC4: 7C980484
	v_add_lshl_u32 v16, v4, v3, 2                              // 000000004DC8: D6470010 020A0704
	s_wait_alu 0xfffe                                          // 000000004DD0: BF88FFFE
	s_add_nc_u64 s[12:13], s[12:13], s[16:17]                  // 000000004DD4: A98C100C
	v_cmp_gt_u64_e64 s4, s[26:27], v[0:1]                      // 000000004DD8: D45C0004 0002001A
	v_add_lshl_u32 v17, v5, v3, 2                              // 000000004DE0: D6470011 020A0705
	s_wait_alu 0xfffd                                          // 000000004DE8: BF88FFFD
	v_cndmask_b32_e64 v4, 4, 0, vcc_lo                         // 000000004DEC: D5010004 01A90084
	v_cmp_gt_u32_e32 vcc_lo, 2, v2                             // 000000004DF4: 7C980482
	v_lshrrev_b32_e32 v9, 3, v0                                // 000000004DF8: 32120083
	v_cmp_gt_u32_e64 s6, 32, v0                                // 000000004DFC: D44C0006 000200A0
	v_cmp_eq_u32_e64 s8, 0, v0                                 // 000000004E04: D44A0008 00020080
	v_add_lshl_u32 v18, v4, v3, 2                              // 000000004E0C: D6470012 020A0704
	s_wait_alu 0xfffd                                          // 000000004E14: BF88FFFD
	v_cndmask_b32_e64 v2, 2, 0, vcc_lo                         // 000000004E18: D5010002 01A90082
	v_cmp_ne_u32_e32 vcc_lo, 31, v3                            // 000000004E20: 7C9A069F
	v_dual_mov_b32 v22, 0 :: v_dual_add_nc_u32 v21, 0x400, v11 // 000000004E24: CA200080 161416FF 00000400
	s_mov_b32 s11, 0                                           // 000000004E30: BE8B0080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004E34: BF870003
	v_add_lshl_u32 v19, v2, v3, 2                              // 000000004E38: D6470013 020A0702
	s_wait_alu 0xfffd                                          // 000000004E40: BF88FFFD
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo                // 000000004E44: D5207C05 01AA0680
	v_add_co_u32 v2, s9, s12, v8                               // 000000004E4C: D7000902 0002100C
	s_wait_alu 0xf1ff                                          // 000000004E54: BF88F1FF
	v_add_co_ci_u32_e64 v3, null, s13, 0, s9                   // 000000004E58: D5207C03 0025000D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004E60: BF870003
	v_lshlrev_b32_e32 v20, 2, v5                               // 000000004E64: 30280A82
	s_lshr_b32 s36, s30, 1                                     // 000000004E68: 8524811E
	s_lshl_b32 s37, s30, 2                                     // 000000004E6C: 8425821E
	s_lshl_b32 s38, s30, 2                                     // 000000004E70: 8426821E
	s_mov_b32 s39, 0xff7fffff                                  // 000000004E74: BEA700FF FF7FFFFF
	s_sub_nc_u64 s[12:13], s[18:19], s[34:35]                  // 000000004E7C: AA0C2212
	s_mov_b32 s10, s11                                         // 000000004E80: BE8A000B
	s_wait_alu 0xfffe                                          // 000000004E84: BF88FFFE
	v_cmp_lt_u64_e64 s9, s[12:13], 64                          // 000000004E88: D4590009 0001800C
	s_and_b32 s9, s9, exec_lo                                  // 000000004E90: 8B097E09
	s_cselect_b32 s40, s12, 64                                 // 000000004E94: 9828C00C
	s_branch 12                                                // 000000004E98: BFA0000C <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x7cc>
	s_wait_alu 0xfffe                                          // 000000004E9C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000004EA0: 8C7E097E
	s_add_co_i32 s10, s10, 1                                   // 000000004EA4: 810A810A
	s_wait_loadcnt_dscnt 0x0                                   // 000000004EA8: BFC80000
	s_wait_alu 0xfffe                                          // 000000004EAC: BF88FFFE
	s_cmp_ge_u32 s10, s40                                      // 000000004EB0: BF09280A
	s_barrier_signal -1                                        // 000000004EB4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004EB8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004EBC: EE0AC07C 00040000 00000000
	s_cbranch_scc1 141                                         // 000000004EC8: BFA2008D <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xa00>
	v_mov_b32_e32 v23, 0                                       // 000000004ECC: 7E2E0280
	s_and_saveexec_b32 s41, s4                                 // 000000004ED0: BEA92004
	s_cbranch_execz 50                                         // 000000004ED4: BFA50032 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x8a0>
	s_add_nc_u64 s[16:17], s[34:35], s[10:11]                  // 000000004ED8: A9900A22
	v_mov_b32_e32 v5, v3                                       // 000000004EDC: 7E0A0303
	s_wait_alu 0xfffe                                          // 000000004EE0: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[24:25]                     // 000000004EE4: AA901810
	v_mov_b32_e32 v7, v1                                       // 000000004EE8: 7E0E0301
	s_wait_alu 0xfffe                                          // 000000004EEC: BF88FFFE
	s_add_nc_u64 s[16:17], s[16:17], s[22:23]                  // 000000004EF0: A9901610
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v4, v2             // 000000004EF4: CA100080 17040102
	s_wait_alu 0xfffe                                          // 000000004EFC: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[26:27]                     // 000000004F00: AA901A10
	v_mov_b32_e32 v6, v0                                       // 000000004F04: 7E0C0300
	s_wait_alu 0xfffe                                          // 000000004F08: BF88FFFE
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000004F0C: 84908210
	s_mov_b32 s42, 0                                           // 000000004F10: BEAA0080
	s_wait_alu 0xfffe                                          // 000000004F14: BF88FFFE
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]                  // 000000004F18: A990100E
	v_lshlrev_b64_e32 v[24:25], 2, v[6:7]                      // 000000004F1C: 3E300C82
	s_wait_alu 0xfffe                                          // 000000004F20: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004F24: BF870121
	v_add_co_u32 v24, vcc_lo, s16, v24                         // 000000004F28: D7006A18 00023010
	s_wait_alu 0xfffd                                          // 000000004F30: BF88FFFD
	v_add_co_ci_u32_e64 v25, null, s17, v25, vcc_lo            // 000000004F34: D5207C19 01AA3211
	v_add_co_u32 v6, vcc_lo, v6, s30                           // 000000004F3C: D7006A06 00003D06
	global_load_b32 v26, v[4:5], off                           // 000000004F44: EE05007C 0000001A 00000004
	global_load_b32 v24, v[24:25], off                         // 000000004F50: EE05007C 00000018 00000018
	s_wait_alu 0xfffd                                          // 000000004F5C: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo                // 000000004F60: D5207C07 01AA0E80
	v_add_co_u32 v4, s9, v4, s37                               // 000000004F68: D7000904 00004B04
	s_wait_alu 0xf1ff                                          // 000000004F70: BF88F1FF
	v_add_co_ci_u32_e64 v5, null, 0, v5, s9                    // 000000004F74: D5207C05 00260A80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004F7C: BF870003
	v_cmp_le_u64_e32 vcc_lo, s[26:27], v[6:7]                  // 000000004F80: 7CB60C1A
	s_or_b32 s42, vcc_lo, s42                                  // 000000004F84: 8C2A2A6A
	s_wait_loadcnt 0x0                                         // 000000004F88: BFC00000
	v_fmac_f32_e32 v23, v26, v24                               // 000000004F8C: 562E311A
	s_wait_alu 0xfffe                                          // 000000004F90: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s42                       // 000000004F94: 917E2A7E
	s_cbranch_execnz 65504                                     // 000000004F98: BFA6FFE0 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x81c>
	s_or_b32 exec_lo, exec_lo, s42                             // 000000004F9C: 8C7E2A7E
	s_wait_alu 0xfffe                                          // 000000004FA0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s41                             // 000000004FA4: 8C7E297E
	ds_bpermute_b32 v4, v16, v23                               // 000000004FA8: DACC0000 04001710
	s_wait_dscnt 0x0                                           // 000000004FB0: BFC60000
	v_add_f32_e32 v4, v23, v4                                  // 000000004FB4: 06080917
	ds_bpermute_b32 v5, v17, v4                                // 000000004FB8: DACC0000 05000411
	s_wait_dscnt 0x0                                           // 000000004FC0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000004FC4: 06080B04
	ds_bpermute_b32 v5, v18, v4                                // 000000004FC8: DACC0000 05000412
	s_wait_dscnt 0x0                                           // 000000004FD0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000004FD4: 06080B04
	ds_bpermute_b32 v5, v19, v4                                // 000000004FD8: DACC0000 05000413
	s_wait_dscnt 0x0                                           // 000000004FE0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000004FE4: 06080B04
	ds_bpermute_b32 v5, v20, v4                                // 000000004FE8: DACC0000 05000414
	s_and_saveexec_b32 s9, s5                                  // 000000004FF0: BE892005
	s_cbranch_execz 4                                          // 000000004FF4: BFA50004 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x908>
	s_wait_dscnt 0x0                                           // 000000004FF8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000004FFC: 06080B04
	ds_store_b32 v9, v4                                        // 000000005000: D8340000 00000409
	s_wait_alu 0xfffe                                          // 000000005008: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000500C: 8C7E097E
	s_wait_dscnt 0x0                                           // 000000005010: BFC60000
	s_barrier_signal -1                                        // 000000005014: BE804EC1
	s_barrier_wait 0xffff                                      // 000000005018: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 00000000501C: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s6                                  // 000000005028: BE892006
	s_cbranch_execz 31                                         // 00000000502C: BFA5001F <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x9ac>
	v_mov_b32_e32 v4, 0                                        // 000000005030: 7E080280
	s_and_saveexec_b32 s16, s7                                 // 000000005034: BE902007
	ds_load_b32 v4, v10                                        // 000000005038: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000005040: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 000000005044: 8C7E107E
	s_wait_dscnt 0x0                                           // 000000005048: BFC60000
	ds_bpermute_b32 v5, v16, v4                                // 00000000504C: DACC0000 05000410
	s_wait_dscnt 0x0                                           // 000000005054: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000005058: 06080B04
	ds_bpermute_b32 v5, v17, v4                                // 00000000505C: DACC0000 05000411
	s_wait_dscnt 0x0                                           // 000000005064: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000005068: 06080B04
	ds_bpermute_b32 v5, v18, v4                                // 00000000506C: DACC0000 05000412
	s_wait_dscnt 0x0                                           // 000000005074: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000005078: 06080B04
	ds_bpermute_b32 v5, v19, v4                                // 00000000507C: DACC0000 05000413
	s_wait_dscnt 0x0                                           // 000000005084: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000005088: 06080B04
	ds_bpermute_b32 v5, v20, v4                                // 00000000508C: DACC0000 05000414
	s_and_b32 exec_lo, exec_lo, s5                             // 000000005094: 8B7E057E
	s_cbranch_execz 4                                          // 000000005098: BFA50004 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x9ac>
	s_wait_dscnt 0x0                                           // 00000000509C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000050A0: 06080B04
	ds_store_b32 v12, v4                                       // 0000000050A4: D8340000 0000040C
	s_wait_alu 0xfffe                                          // 0000000050AC: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000050B0: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000050B4: BFC80000
	s_barrier_signal -1                                        // 0000000050B8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000050BC: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000050C0: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s8                                  // 0000000050CC: BE892008
	s_cbranch_execz 65394                                      // 0000000050D0: BFA5FF72 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x79c>
	ds_load_b32 v4, v12                                        // 0000000050D4: D8D80000 0400000C
	s_lshl_b32 s16, s10, 2                                     // 0000000050DC: 8410820A
	s_wait_dscnt 0x0                                           // 0000000050E0: BFC60000
	s_wait_kmcnt 0x0                                           // 0000000050E4: BFC70000
	s_wait_alu 0xfffe                                          // 0000000050E8: BF88FFFE
	v_dual_mov_b32 v5, s16 :: v_dual_mul_f32 v4, s31, v4       // 0000000050EC: CA060010 0504081F
	ds_store_b32 v5, v4 offset:1024                            // 0000000050F4: D8340400 00000405
	s_branch 65383                                             // 0000000050FC: BFA0FF67 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x79c>
	v_cmp_gt_u32_e64 s9, s40, v0                               // 000000005100: D44C0009 00020028
	v_mov_b32_e32 v4, 0xff7fffff                               // 000000005108: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s16, s9                                 // 000000005110: BE902009
	s_cbranch_execz 24                                         // 000000005114: BFA50018 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xa78>
	v_dual_mov_b32 v4, 0xff7fffff :: v_dual_mov_b32 v5, v21    // 000000005118: CA1000FF 04040115 FF7FFFFF
	v_mov_b32_e32 v6, v0                                       // 000000005124: 7E0C0300
	s_mov_b32 s17, 0                                           // 000000005128: BE910080
	ds_load_b32 v7, v5                                         // 00000000512C: D8D80000 07000005
	v_add_nc_u32_e32 v6, s30, v6                               // 000000005134: 4A0C0C1E
	v_add_nc_u32_e32 v5, s38, v5                               // 000000005138: 4A0A0A26
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000513C: BF870002
	v_cmp_le_u32_e32 vcc_lo, s40, v6                           // 000000005140: 7C960C28
	s_wait_alu 0xfffe                                          // 000000005144: BF88FFFE
	s_or_b32 s17, vcc_lo, s17                                  // 000000005148: 8C11116A
	s_wait_dscnt 0x0                                           // 00000000514C: BFC60000
	v_cmp_gt_f32_e64 s10, v7, v4                               // 000000005150: D414000A 00020907
	s_wait_alu 0xf1ff                                          // 000000005158: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000515C: BF870001
	v_cndmask_b32_e64 v4, v4, v7, s10                          // 000000005160: D5010004 002A0F04
	s_wait_alu 0xfffe                                          // 000000005168: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s17                       // 00000000516C: 917E117E
	s_cbranch_execnz 65518                                     // 000000005170: BFA6FFEE <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xa2c>
	s_or_b32 exec_lo, exec_lo, s17                             // 000000005174: 8C7E117E
	s_wait_alu 0xfffe                                          // 000000005178: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 00000000517C: 8C7E107E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005180: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000005184: 916A217E
	s_mov_b32 s10, s36                                         // 000000005188: BE8A0024
	ds_store_b32 v11, v4                                       // 00000000518C: D8340000 0000040B
	s_wait_loadcnt_dscnt 0x0                                   // 000000005194: BFC80000
	s_barrier_signal -1                                        // 000000005198: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000519C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000051A0: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 0000000051AC: BF88FFFE
	s_cbranch_vccz 274                                         // 0000000051B0: BFA30112 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xefc>
	s_and_saveexec_b32 s10, s8                                 // 0000000051B4: BE8A2008
	s_cbranch_execz 53                                         // 0000000051B8: BFA50035 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xb90>
	ds_load_b32 v6, v12                                        // 0000000051BC: D8D80000 0600000C
	s_wait_dscnt 0x0                                           // 0000000051C4: BFC60000
	v_readfirstlane_b32 s16, v6                                // 0000000051C8: 7E200506
	s_cmp_gt_f32 s39, s16                                      // 0000000051CC: BF441027
	s_cselect_b32 s16, s39, s16                                // 0000000051D0: 98101027
	s_wait_alu 0xfffe                                          // 0000000051D4: BF88FFFE
	s_sub_f32 s17, s39, s16                                    // 0000000051D8: A0911027
	v_mov_b32_e32 v5, s16                                      // 0000000051DC: 7E0A0210
	s_wait_alu 0xfffe                                          // 0000000051E0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000051E4: BF870529
	s_mul_f32 s41, s17, 0x3fb8aa3b                             // 0000000051E8: A229FF11 3FB8AA3B
	s_wait_alu 0xfffe                                          // 0000000051F0: BF88FFFE
	s_xor_b32 s42, s41, 0x80000000                             // 0000000051F4: 8D2AFF29 80000000
	s_rndne_f32 s43, s41                                       // 0000000051FC: BEAB6329
	s_wait_alu 0xfffe                                          // 000000005200: BF88FFFE
	s_fmamk_f32 s42, s17, 0x3fb8aa3b, s42                      // 000000005204: A32A2A11 3FB8AA3B
	s_cmp_nlt_f32 s17, 0xc2ce8ed0                              // 00000000520C: BF4EFF11 C2CE8ED0
	s_sub_f32 s41, s41, s43                                    // 000000005214: A0A92B29
	s_wait_alu 0xfffe                                          // 000000005218: BF88FFFE
	s_fmamk_f32 s42, s17, 0x32a5705f, s42                      // 00000000521C: A32A2A11 32A5705F
	s_cselect_b32 vcc_lo, -1, 0                                // 000000005224: 986A80C1
	s_cmp_ngt_f32 s17, 0x42b17218                              // 000000005228: BF4BFF11 42B17218
	s_wait_alu 0xfffe                                          // 000000005230: BF88FFFE
	s_add_f32 s41, s41, s42                                    // 000000005234: A0292A29
	s_cvt_i32_f32 s42, s43                                     // 000000005238: BEAA662B
	s_wait_alu 0xfffe                                          // 00000000523C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)// 000000005240: BF8702A9
	v_s_exp_f32 s41, s41                                       // 000000005244: D6800029 00000029
	s_wait_alu 0xf1ff                                          // 00000000524C: BF88F1FF
	v_ldexp_f32 v4, s41, s42                                   // 000000005250: D71C0004 00005429
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000005258: BF8700C1
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 00000000525C: 02080880
	s_cselect_b32 vcc_lo, -1, 0                                // 000000005260: 986A80C1
	s_cmp_nle_f32 s39, 0xff61b1e6                              // 000000005264: BF4CFF27 FF61B1E6
	s_wait_alu 0xfffe                                          // 00000000526C: BF88FFFE
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo               // 000000005270: 020808FF 7F800000
	s_cselect_b32 vcc_lo, -1, 0                                // 000000005278: 986A80C1
	s_wait_alu 0xfffe                                          // 00000000527C: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005280: BF870001
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000005284: 02080880
	ds_store_b96 v12, v[4:6] offset:1280                       // 000000005288: DB780500 0000040C
	s_wait_alu 0xfffe                                          // 000000005290: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000005294: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000005298: BFC80000
	s_barrier_signal -1                                        // 00000000529C: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000052A0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000052A4: EE0AC07C 00040000 00000000
	ds_load_b32 v4, v12 offset:1284                            // 0000000052B0: D8D80504 0400000C
	s_wait_dscnt 0x0                                           // 0000000052B8: BFC60000
	v_readfirstlane_b32 s39, v4                                // 0000000052BC: 7E4E0504
	v_mov_b32_e32 v4, 0                                        // 0000000052C0: 7E080280
	s_and_saveexec_b32 s10, s9                                 // 0000000052C4: BE8A2009
	s_cbranch_execz 47                                         // 0000000052C8: BFA5002F <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xc88>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, v21             // 0000000052CC: CA100080 04040115
	v_mov_b32_e32 v6, v0                                       // 0000000052D4: 7E0C0300
	s_mov_b32 s9, 0                                            // 0000000052D8: BE890080
	ds_load_b32 v7, v5                                         // 0000000052DC: D8D80000 07000005
	s_wait_dscnt 0x0                                           // 0000000052E4: BFC60000
	v_dual_subrev_f32 v7, s39, v7 :: v_dual_add_nc_u32 v6, s30, v6// 0000000052E8: C9A00E27 07060C1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000052F0: BF870121
	v_mul_f32_e32 v23, 0x3fb8aa3b, v7                          // 0000000052F4: 102E0EFF 3FB8AA3B
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v7                   // 0000000052FC: 7C360EFF C2CE8ED0
	v_fma_f32 v24, 0x3fb8aa3b, v7, -v23                        // 000000005304: D6130018 845E0EFF 3FB8AA3B
	v_rndne_f32_e32 v25, v23                                   // 000000005310: 7E324717
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005314: BF870091
	v_dual_fmac_f32 v24, 0x32a5705f, v7 :: v_dual_sub_f32 v23, v23, v25// 000000005318: C80A0EFF 18163317 32A5705F
	v_add_f32_e32 v23, v23, v24                                // 000000005324: 062E3117
	v_cvt_i32_f32_e32 v24, v25                                 // 000000005328: 7E301119
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000000532C: BF870292
	v_exp_f32_e32 v23, v23                                     // 000000005330: 7E2E4B17
	v_ldexp_f32 v23, v23, v24                                  // 000000005334: D71C0017 00023117
	s_wait_alu 0xfffd                                          // 00000000533C: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000005340: BF870131
	v_cndmask_b32_e32 v23, 0, v23, vcc_lo                      // 000000005344: 022E2E80
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v7                   // 000000005348: 7C3C0EFF 42B17218
	s_wait_alu 0xfffd                                          // 000000005350: BF88FFFD
	v_cndmask_b32_e32 v7, 0x7f800000, v23, vcc_lo              // 000000005354: 020E2EFF 7F800000
	v_cmp_le_u32_e32 vcc_lo, s40, v6                           // 00000000535C: 7C960C28
	ds_store_b32 v5, v7                                        // 000000005360: D8340000 00000705
	v_dual_add_f32 v4, v4, v7 :: v_dual_add_nc_u32 v5, s38, v5 // 000000005368: C9200F04 04040A26
	s_wait_alu 0xfffe                                          // 000000005370: BF88FFFE
	s_or_b32 s9, vcc_lo, s9                                    // 000000005374: 8C09096A
	s_wait_alu 0xfffe                                          // 000000005378: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s9                        // 00000000537C: 917E097E
	s_cbranch_execnz 65494                                     // 000000005380: BFA6FFD6 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xbdc>
	s_or_b32 exec_lo, exec_lo, s9                              // 000000005384: 8C7E097E
	s_wait_alu 0xfffe                                          // 000000005388: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 00000000538C: 8C7E0A7E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005390: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000005394: 916A217E
	s_mov_b32 s9, s36                                          // 000000005398: BE890024
	ds_store_b32 v11, v4                                       // 00000000539C: D8340000 0000040B
	s_wait_loadcnt_dscnt 0x0                                   // 0000000053A4: BFC80000
	s_barrier_signal -1                                        // 0000000053A8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000053AC: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000053B0: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 0000000053BC: BF88FFFE
	s_cbranch_vccz 173                                         // 0000000053C0: BFA300AD <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xf78>
	s_and_saveexec_b32 s9, s8                                  // 0000000053C4: BE892008
	s_cbranch_execz 5                                          // 0000000053C8: BFA50005 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xce0>
	ds_load_b32 v4, v12                                        // 0000000053CC: D8D80000 0400000C
	s_wait_dscnt 0x0                                           // 0000000053D4: BFC60000
	ds_store_b32 v12, v4 offset:1292                           // 0000000053D8: D834050C 0000040C
	s_wait_alu 0xfffe                                          // 0000000053E0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000053E4: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000053E8: BFC80000
	s_barrier_signal -1                                        // 0000000053EC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000053F0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000053F4: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s3                                  // 000000005400: BE892003
	s_wait_alu 0xfffe                                          // 000000005404: BF88FFFE
	s_xor_b32 s9, exec_lo, s9                                  // 000000005408: 8D09097E
	ds_load_b32 v5, v12 offset:1280                            // 00000000540C: D8D80500 0500000C
	s_wait_alu 0xfffe                                          // 000000005414: BF88FFFE
	s_and_not1_saveexec_b32 s9, s9                             // 000000005418: BE893009
	s_cbranch_execz 213                                        // 00000000541C: BFA500D5 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1074>
	v_cmp_lt_u64_e64 s10, s[12:13], 4                          // 000000005420: D459000A 0001080C
	v_mov_b32_e32 v4, 0                                        // 000000005428: 7E080280
	s_max_u32 s13, s40, 1                                      // 00000000542C: 8A8D8128
	s_and_b32 vcc_lo, exec_lo, s10                             // 000000005430: 8B6A0A7E
	s_wait_alu 0xfffe                                          // 000000005434: BF88FFFE
	s_cbranch_vccnz 159                                        // 000000005438: BFA4009F <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xfb8>
	s_and_b32 s12, s13, 0x7c                                   // 00000000543C: 8B0CFF0D 0000007C
	s_mov_b32 s10, 0                                           // 000000005444: BE8A0080
	s_movk_i32 s16, 0x400                                      // 000000005448: B0100400
	s_wait_alu 0xfffe                                          // 00000000544C: BF88FFFE
	s_add_nc_u64 s[40:41], s[34:35], s[10:11]                  // 000000005450: A9A80A22
	s_or_b32 s42, s10, 1                                       // 000000005454: 8C2A810A
	s_mov_b32 s43, s11                                         // 000000005458: BEAB000B
	s_wait_alu 0xfffe                                          // 00000000545C: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[24:25]                     // 000000005460: AAA81828
	s_add_nc_u64 s[42:43], s[34:35], s[42:43]                  // 000000005464: A9AA2A22
	s_wait_alu 0xfffe                                          // 000000005468: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[22:23]                  // 00000000546C: A9A81628
	s_mul_u64 s[42:43], s[42:43], s[24:25]                     // 000000005470: AAAA182A
	s_wait_alu 0xfffe                                          // 000000005474: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 000000005478: AAA81428
	s_add_nc_u64 s[42:43], s[42:43], s[22:23]                  // 00000000547C: A9AA162A
	s_wait_alu 0xfffe                                          // 000000005480: BF88FFFE
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000005484: 84A88228
	s_or_b32 s44, s10, 2                                       // 000000005488: 8C2C820A
	s_mov_b32 s45, s11                                         // 00000000548C: BEAD000B
	s_mul_u64 s[42:43], s[42:43], s[20:21]                     // 000000005490: AAAA142A
	s_wait_dscnt 0x0                                           // 000000005494: BFC60000
	s_wait_alu 0xfffe                                          // 000000005498: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v14, s40                          // 00000000549C: D7006A05 0000510E
	s_or_b32 s46, s10, 3                                       // 0000000054A4: 8C2E830A
	s_mov_b32 s47, s11                                         // 0000000054A8: BEAF000B
	s_add_nc_u64 s[44:45], s[34:35], s[44:45]                  // 0000000054AC: A9AC2C22
	s_wait_alu 0xfffd                                          // 0000000054B0: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s41, v15, vcc_lo             // 0000000054B4: D5207C06 01AA1E29
	s_lshl_b64 s[40:41], s[42:43], 2                           // 0000000054BC: 84A8822A
	s_add_nc_u64 s[46:47], s[34:35], s[46:47]                  // 0000000054C0: A9AE2E22
	s_wait_alu 0xfffe                                          // 0000000054C4: BF88FFFE
	s_mul_u64 s[44:45], s[44:45], s[24:25]                     // 0000000054C8: AAAC182C
	v_add_co_u32 v23, vcc_lo, v14, s40                         // 0000000054CC: D7006A17 0000510E
	s_mul_u64 s[46:47], s[46:47], s[24:25]                     // 0000000054D4: AAAE182E
	s_wait_alu 0xfffe                                          // 0000000054D8: BF88FFFE
	s_add_nc_u64 s[44:45], s[44:45], s[22:23]                  // 0000000054DC: A9AC162C
	s_wait_alu 0xfffd                                          // 0000000054E0: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s41, v15, vcc_lo            // 0000000054E4: D5207C18 01AA1E29
	s_add_nc_u64 s[46:47], s[46:47], s[22:23]                  // 0000000054EC: A9AE162E
	s_wait_alu 0xfffe                                          // 0000000054F0: BF88FFFE
	s_mul_u64 s[44:45], s[44:45], s[20:21]                     // 0000000054F4: AAAC142C
	s_clause 0x1                                               // 0000000054F8: BF850001
	global_load_b32 v7, v[5:6], off                            // 0000000054FC: EE05007C 00000007 00000005
	global_load_b32 v27, v[23:24], off                         // 000000005508: EE05007C 0000001B 00000017
	s_mul_u64 s[46:47], s[46:47], s[20:21]                     // 000000005514: AAAE142E
	s_wait_alu 0xfffe                                          // 000000005518: BF88FFFE
	s_lshl_b64 s[42:43], s[44:45], 2                           // 00000000551C: 84AA822C
	s_lshl_b64 s[44:45], s[46:47], 2                           // 000000005520: 84AC822E
	s_wait_alu 0xfffe                                          // 000000005524: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v14, s42                          // 000000005528: D7006A05 0000550E
	s_wait_alu 0xfffd                                          // 000000005530: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s43, v15, vcc_lo             // 000000005534: D5207C06 01AA1E2B
	v_add_co_u32 v23, vcc_lo, v14, s44                         // 00000000553C: D7006A17 0000590E
	s_wait_alu 0xfffd                                          // 000000005544: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s45, v15, vcc_lo            // 000000005548: D5207C18 01AA1E2D
	s_clause 0x1                                               // 000000005550: BF850001
	global_load_b32 v5, v[5:6], off                            // 000000005554: EE05007C 00000005 00000005
	global_load_b32 v6, v[23:24], off                          // 000000005560: EE05007C 00000006 00000017
	v_mov_b32_e32 v23, s16                                     // 00000000556C: 7E2E0210
	s_add_co_i32 s10, s10, 4                                   // 000000005570: 810A840A
	s_add_co_i32 s16, s16, 16                                  // 000000005574: 81109010
	s_wait_alu 0xfffe                                          // 000000005578: BF88FFFE
	s_cmp_eq_u32 s12, s10                                      // 00000000557C: BF060A0C
	ds_load_b128 v[23:26], v23                                 // 000000005580: DBFC0000 17000017
	s_wait_loadcnt_dscnt 0x300                                 // 000000005588: BFC80300
	v_fmac_f32_e32 v4, v23, v7                                 // 00000000558C: 56080F17
	s_wait_loadcnt 0x2                                         // 000000005590: BFC00002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005594: BF8700A1
	v_fmac_f32_e32 v4, v24, v27                                // 000000005598: 56083718
	s_wait_loadcnt 0x1                                         // 00000000559C: BFC00001
	v_fmac_f32_e32 v4, v25, v5                                 // 0000000055A0: 56080B19
	s_wait_loadcnt 0x0                                         // 0000000055A4: BFC00000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000055A8: BF870001
	v_fmac_f32_e32 v4, v26, v6                                 // 0000000055AC: 56080D1A
	s_cbranch_scc0 65446                                       // 0000000055B0: BFA1FFA6 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xd4c>
	s_and_b32 s13, s13, 3                                      // 0000000055B4: 8B0D830D
	s_wait_alu 0xfffe                                          // 0000000055B8: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 0000000055BC: BF06800D
	s_cbranch_scc0 66                                          // 0000000055C0: BFA10042 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xfcc>
	s_branch 100                                               // 0000000055C4: BFA00064 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1058>
	s_wait_alu 0xfffe                                          // 0000000055C8: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 0000000055CC: 8C7E107E
	s_lshr_b32 s16, s10, 1                                     // 0000000055D0: 8510810A
	s_cmp_gt_u32 s10, 1                                        // 0000000055D4: BF08810A
	s_wait_alu 0xfffe                                          // 0000000055D8: BF88FFFE
	s_mov_b32 s10, s16                                         // 0000000055DC: BE8A0010
	s_wait_loadcnt_dscnt 0x0                                   // 0000000055E0: BFC80000
	s_barrier_signal -1                                        // 0000000055E4: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000055E8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000055EC: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65262                                       // 0000000055F8: BFA1FEEE <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xab4>
	s_mov_b32 s16, exec_lo                                     // 0000000055FC: BE90007E
	s_wait_alu 0xfffe                                          // 000000005600: BF88FFFE
	v_cmpx_gt_u32_e64 s10, v0                                  // 000000005604: D4CC007E 0002000A
	s_cbranch_execz 65518                                      // 00000000560C: BFA5FFEE <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xec8>
	v_lshl_add_u32 v4, s10, 2, v11                             // 000000005610: D6460004 042D040A
	ds_load_b32 v5, v11                                        // 000000005618: D8D80000 0500000B
	ds_load_b32 v4, v4                                         // 000000005620: D8D80000 04000004
	s_wait_dscnt 0x0                                           // 000000005628: BFC60000
	v_cmp_gt_f32_e32 vcc_lo, v5, v4                            // 00000000562C: 7C280905
	s_wait_alu 0xfffd                                          // 000000005630: BF88FFFD
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000005634: 02080B04
	ds_store_b32 v11, v4                                       // 000000005638: D8340000 0000040B
	s_branch 65505                                             // 000000005640: BFA0FFE1 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xec8>
	s_wait_alu 0xfffe                                          // 000000005644: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000005648: 8C7E0A7E
	s_lshr_b32 s10, s9, 1                                      // 00000000564C: 850A8109
	s_cmp_gt_u32 s9, 1                                         // 000000005650: BF088109
	s_wait_alu 0xfffe                                          // 000000005654: BF88FFFE
	s_mov_b32 s9, s10                                          // 000000005658: BE89000A
	s_wait_loadcnt_dscnt 0x0                                   // 00000000565C: BFC80000
	s_barrier_signal -1                                        // 000000005660: BE804EC1
	s_barrier_wait 0xffff                                      // 000000005664: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000005668: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65363                                       // 000000005674: BFA1FF53 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xcc4>
	s_mov_b32 s10, exec_lo                                     // 000000005678: BE8A007E
	s_wait_alu 0xfffe                                          // 00000000567C: BF88FFFE
	v_cmpx_gt_u32_e64 s9, v0                                   // 000000005680: D4CC007E 00020009
	s_cbranch_execz 65518                                      // 000000005688: BFA5FFEE <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xf44>
	v_lshl_add_u32 v4, s9, 2, v11                              // 00000000568C: D6460004 042D0409
	ds_load_b32 v4, v4                                         // 000000005694: D8D80000 04000004
	ds_load_b32 v5, v11                                        // 00000000569C: D8D80000 0500000B
	s_wait_dscnt 0x0                                           // 0000000056A4: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000056A8: 06080B04
	ds_store_b32 v11, v4                                       // 0000000056AC: D8340000 0000040B
	s_branch 65507                                             // 0000000056B4: BFA0FFE3 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xf44>
	s_mov_b32 s12, 0                                           // 0000000056B8: BE8C0080
	s_and_b32 s13, s13, 3                                      // 0000000056BC: 8B0D830D
	s_wait_alu 0xfffe                                          // 0000000056C0: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 0000000056C4: BF06800D
	s_cbranch_scc1 35                                          // 0000000056C8: BFA20023 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1058>
	s_lshl_b32 s10, s12, 2                                     // 0000000056CC: 840A820C
	s_wait_alu 0xfffe                                          // 0000000056D0: BF88FFFE
	s_or_b32 s16, s10, 0x400                                   // 0000000056D4: 8C10FF0A 00000400
	s_mov_b32 s10, s12                                         // 0000000056DC: BE8A000C
	s_wait_alu 0xfffe                                          // 0000000056E0: BF88FFFE
	s_add_nc_u64 s[40:41], s[34:35], s[10:11]                  // 0000000056E4: A9A80A22
	s_add_co_i32 s13, s13, -1                                  // 0000000056E8: 810DC10D
	s_wait_alu 0xfffe                                          // 0000000056EC: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[24:25]                     // 0000000056F0: AAA81828
	s_add_co_i32 s10, s10, 1                                   // 0000000056F4: 810A810A
	s_wait_alu 0xfffe                                          // 0000000056F8: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[22:23]                  // 0000000056FC: A9A81628
	s_wait_alu 0xfffe                                          // 000000005700: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 000000005704: AAA81428
	s_wait_alu 0xfffe                                          // 000000005708: BF88FFFE
	s_lshl_b64 s[40:41], s[40:41], 2                           // 00000000570C: 84A88228
	s_wait_dscnt 0x0                                           // 000000005710: BFC60000
	s_wait_alu 0xfffe                                          // 000000005714: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v14, s40                          // 000000005718: D7006A05 0000510E
	s_wait_alu 0xfffd                                          // 000000005720: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s41, v15, vcc_lo             // 000000005724: D5207C06 01AA1E29
	global_load_b32 v5, v[5:6], off                            // 00000000572C: EE05007C 00000005 00000005
	v_mov_b32_e32 v6, s16                                      // 000000005738: 7E0C0210
	s_add_co_i32 s16, s16, 4                                   // 00000000573C: 81108410
	s_cmp_lg_u32 s13, 0                                        // 000000005740: BF07800D
	ds_load_b32 v6, v6                                         // 000000005744: D8D80000 06000006
	s_wait_loadcnt_dscnt 0x0                                   // 00000000574C: BFC80000
	v_fmac_f32_e32 v4, v6, v5                                  // 000000005750: 56080B06
	s_cbranch_scc1 65506                                       // 000000005754: BFA2FFE2 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0xfe0>
	s_wait_dscnt 0x0                                           // 000000005758: BFC60000
	ds_load_b32 v5, v12 offset:1280                            // 00000000575C: D8D80500 0500000C
	s_wait_dscnt 0x0                                           // 000000005764: BFC60000
	v_fmac_f32_e32 v4, v13, v5                                 // 000000005768: 56080B0D
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000576C: BF870001
	v_mov_b32_e32 v13, v4                                      // 000000005770: 7E1A0304
	s_wait_alu 0xfffe                                          // 000000005774: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000005778: 8C7E097E
	ds_load_b32 v4, v12 offset:1292                            // 00000000577C: D8D8050C 0400000C
	s_add_nc_u64 s[34:35], s[34:35], 64                        // 000000005784: A9A2C022
	s_wait_loadcnt_dscnt 0x0                                   // 000000005788: BFC80000
	s_wait_alu 0xfffe                                          // 00000000578C: BF88FFFE
	v_cmp_ge_u64_e64 s9, s[34:35], s[18:19]                    // 000000005790: D45E0009 00002422
	s_barrier_signal -1                                        // 000000005798: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000579C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000057A0: EE0AC07C 00040000 00000000
	s_and_b32 vcc_lo, exec_lo, s9                              // 0000000057AC: 8B6A097E
	v_fmac_f32_e32 v4, v22, v5                                 // 0000000057B0: 56080B16
	s_wait_alu 0xfffe                                          // 0000000057B4: BF88FFFE
	s_cbranch_vccnz 3                                          // 0000000057B8: BFA40003 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x10c8>
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000057BC: BF870001
	v_mov_b32_e32 v22, v4                                      // 0000000057C0: 7E2C0304
	s_branch 64941                                             // 0000000057C4: BFA0FDAD <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x77c>
	s_and_saveexec_b32 s3, s2                                  // 0000000057C8: BE832002
	s_cbranch_execz 34                                         // 0000000057CC: BFA50022 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x1158>
	v_div_scale_f32 v0, null, v4, v4, v13                      // 0000000057D0: D6FC7C00 04360904
	s_load_b64 s[0:1], s[0:1], 0x50                            // 0000000057D8: F4002000 F8000050
	s_mul_u64 s[2:3], s[20:21], s[28:29]                       // 0000000057E0: AA821C14
	s_wait_alu 0xfffe                                          // 0000000057E4: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000057E8: 84828202
	v_rcp_f32_e32 v1, v0                                       // 0000000057EC: 7E025500
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000057F0: BF870095
	v_fma_f32 v2, -v0, v1, 1.0                                 // 0000000057F4: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 0000000057FC: 56020302
	v_div_scale_f32 v2, vcc_lo, v13, v4, v13                   // 000000005800: D6FC6A02 0436090D
	s_wait_kmcnt 0x0                                           // 000000005808: BFC70000
	s_wait_alu 0xfffe                                          // 00000000580C: BF88FFFE
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000005810: A9800200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005814: BF870091
	v_mul_f32_e32 v3, v2, v1                                   // 000000005818: 10060302
	v_fma_f32 v5, -v0, v3, v2                                  // 00000000581C: D6130005 240A0700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005824: BF870091
	v_fmac_f32_e32 v3, v5, v1                                  // 000000005828: 56060305
	v_fma_f32 v0, -v0, v3, v2                                  // 00000000582C: D6130000 240A0700
	s_wait_alu 0xfffd                                          // 000000005834: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005838: BF870091
	v_div_fmas_f32 v0, v0, v1, v3                              // 00000000583C: D6370000 040E0300
	v_div_fixup_f32 v0, v0, v4, v13                            // 000000005844: D6270000 04360900
	global_store_b32 v8, v0, s[0:1]                            // 00000000584C: EE068000 00000000 00000008
	s_endpgm                                                   // 000000005858: BFB00000
	s_branch 64618                                             // 00000000585C: BFA0FC6A <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x308>
	s_branch 64785                                             // 000000005860: BFA0FD11 <ullm_sq8_0_flash2_qk_wave32_prototype_kernel+0x5a8>
	s_nop 0                                                    // 000000005864: BF800000
	s_nop 0                                                    // 000000005868: BF800000
	s_nop 0                                                    // 00000000586C: BF800000
	s_nop 0                                                    // 000000005870: BF800000
	s_nop 0                                                    // 000000005874: BF800000
	s_nop 0                                                    // 000000005878: BF800000
	s_nop 0                                                    // 00000000587C: BF800000
	s_nop 0                                                    // 000000005880: BF800000
	s_nop 0                                                    // 000000005884: BF800000
	s_nop 0                                                    // 000000005888: BF800000
	s_nop 0                                                    // 00000000588C: BF800000
	s_nop 0                                                    // 000000005890: BF800000
	s_nop 0                                                    // 000000005894: BF800000
	s_nop 0                                                    // 000000005898: BF800000
	s_nop 0                                                    // 00000000589C: BF800000
	s_nop 0                                                    // 0000000058A0: BF800000
	s_nop 0                                                    // 0000000058A4: BF800000
	s_nop 0                                                    // 0000000058A8: BF800000
	s_nop 0                                                    // 0000000058AC: BF800000
	s_nop 0                                                    // 0000000058B0: BF800000
	s_nop 0                                                    // 0000000058B4: BF800000
	s_nop 0                                                    // 0000000058B8: BF800000
	s_nop 0                                                    // 0000000058BC: BF800000
	s_nop 0                                                    // 0000000058C0: BF800000
	s_nop 0                                                    // 0000000058C4: BF800000
	s_nop 0                                                    // 0000000058C8: BF800000
	s_nop 0                                                    // 0000000058CC: BF800000
	s_nop 0                                                    // 0000000058D0: BF800000
	s_nop 0                                                    // 0000000058D4: BF800000
	s_nop 0                                                    // 0000000058D8: BF800000
	s_nop 0                                                    // 0000000058DC: BF800000
	s_nop 0                                                    // 0000000058E0: BF800000
	s_nop 0                                                    // 0000000058E4: BF800000
	s_nop 0                                                    // 0000000058E8: BF800000
	s_nop 0                                                    // 0000000058EC: BF800000
	s_nop 0                                                    // 0000000058F0: BF800000
	s_nop 0                                                    // 0000000058F4: BF800000
	s_nop 0                                                    // 0000000058F8: BF800000
	s_nop 0                                                    // 0000000058FC: BF800000

0000000000005900 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel>:
	s_load_b512 s[12:27], s[0:1], 0x0                          // 000000005900: F4008300 F8000000
	s_mov_b32 s31, 0                                           // 000000005908: BE9F0080
	s_mov_b32 s28, ttmp9                                       // 00000000590C: BE9C0075
	s_mov_b32 s29, s31                                         // 000000005910: BE9D001F
	s_wait_kmcnt 0x0                                           // 000000005914: BFC70000
	s_mul_u64 s[2:3], s[22:23], s[20:21]                       // 000000005918: AA821416
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000591C: BF870009
	v_cmp_le_u64_e64 s2, s[2:3], s[28:29]                      // 000000005920: D45B0002 00003802
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000005928: 8B6A027E
	s_cbranch_vccnz 1177                                       // 00000000592C: BFA40499 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1294>
	s_clause 0x1                                               // 000000005930: BF850001
	s_load_b32 s2, s[0:1], 0x64                                // 000000005934: F4000080 F8000064
	s_load_b64 s[20:21], s[0:1], 0x40                          // 00000000593C: F4002500 F8000040
	s_wait_kmcnt 0x0                                           // 000000005944: BFC70000
	s_and_b32 s30, s2, 0xffff                                  // 000000005948: 8B1EFF02 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005950: BF870009
	v_cmp_gt_u64_e64 s2, s[20:21], s[30:31]                    // 000000005954: D45C0002 00003C14
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000595C: 8B6A027E
	s_cbranch_vccnz 1164                                       // 000000005960: BFA4048C <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1294>
	v_cmp_lt_u64_e64 s2, s[28:29], s[22:23]                    // 000000005964: D4590002 00002C1C
	s_and_b32 vcc_lo, exec_lo, s2                              // 00000000596C: 8B6A027E
	s_mov_b64 s[2:3], 0                                        // 000000005970: BE820180
	s_cbranch_vccnz 32                                         // 000000005974: BFA40020 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xf8>
	v_cvt_f32_u32_e32 v1, s22                                  // 000000005978: 7E020C16
	s_sub_co_i32 s3, 0, s22                                    // 00000000597C: 81831680
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005980: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000005984: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000005988: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005990: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000005994: 7E020F01
	v_readfirstlane_b32 s2, v1                                 // 000000005998: 7E040501
	s_wait_alu 0xfffe                                          // 00000000599C: BF88FFFE
	s_mul_i32 s3, s3, s2                                       // 0000000059A0: 96030203
	s_wait_alu 0xfffe                                          // 0000000059A4: BF88FFFE
	s_mul_hi_u32 s3, s2, s3                                    // 0000000059A8: 96830302
	s_wait_alu 0xfffe                                          // 0000000059AC: BF88FFFE
	s_add_co_i32 s2, s2, s3                                    // 0000000059B0: 81020302
	s_wait_alu 0xfffe                                          // 0000000059B4: BF88FFFE
	s_mul_hi_u32 s2, s28, s2                                   // 0000000059B8: 9682021C
	s_wait_alu 0xfffe                                          // 0000000059BC: BF88FFFE
	s_mul_i32 s3, s2, s22                                      // 0000000059C0: 96031602
	s_add_co_i32 s4, s2, 1                                     // 0000000059C4: 81048102
	s_wait_alu 0xfffe                                          // 0000000059C8: BF88FFFE
	s_sub_co_i32 s3, s28, s3                                   // 0000000059CC: 8183031C
	s_wait_alu 0xfffe                                          // 0000000059D0: BF88FFFE
	s_sub_co_i32 s5, s3, s22                                   // 0000000059D4: 81851603
	s_cmp_ge_u32 s3, s22                                       // 0000000059D8: BF091603
	s_cselect_b32 s2, s4, s2                                   // 0000000059DC: 98020204
	s_cselect_b32 s3, s5, s3                                   // 0000000059E0: 98030305
	s_wait_alu 0xfffe                                          // 0000000059E4: BF88FFFE
	s_add_co_i32 s4, s2, 1                                     // 0000000059E8: 81048102
	s_cmp_ge_u32 s3, s22                                       // 0000000059EC: BF091603
	s_mov_b32 s3, 0                                            // 0000000059F0: BE830080
	s_cselect_b32 s2, s4, s2                                   // 0000000059F4: 98020204
	s_or_b64 s[6:7], s[22:23], s[24:25]                        // 0000000059F8: 8C861816
	s_mov_b32 s6, 0                                            // 0000000059FC: BE860080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005A00: BF870009
	s_cmp_lg_u64 s[6:7], 0                                     // 000000005A04: BF118006
	s_cbranch_scc0 1123                                        // 000000005A08: BFA10463 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1298>
	s_cvt_f32_u32 s4, s24                                      // 000000005A0C: BE846518
	s_cvt_f32_u32 s5, s25                                      // 000000005A10: BE856519
	s_sub_nc_u64 s[8:9], 0, s[24:25]                           // 000000005A14: AA081880
	s_mov_b32 s11, s6                                          // 000000005A18: BE8B0006
	s_mov_b32 s37, s6                                          // 000000005A1C: BEA50006
	s_fmamk_f32 s4, s5, 0x4f800000, s4                         // 000000005A20: A3040405 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005A28: BF87029B
	v_s_rcp_f32 s4, s4                                         // 000000005A2C: D6840004 00000004
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000005A34: A204FF04 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000005A3C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000005A40: BF87052A
	s_mul_f32 s5, s4, 0x2f800000                               // 000000005A44: A205FF04 2F800000
	s_wait_alu 0xfffe                                          // 000000005A4C: BF88FFFE
	s_trunc_f32 s5, s5                                         // 000000005A50: BE856205
	s_wait_alu 0xfffe                                          // 000000005A54: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000005A58: BF8704BA
	s_fmamk_f32 s4, s5, 0xcf800000, s4                         // 000000005A5C: A3040405 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000005A64: BE856705
	s_wait_alu 0xfffe                                          // 000000005A68: BF88FFFE
	s_cvt_u32_f32 s4, s4                                       // 000000005A6C: BE846704
	s_wait_alu 0xfffe                                          // 000000005A70: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000005A74: BF87049A
	s_mul_u64 s[34:35], s[8:9], s[4:5]                         // 000000005A78: AAA20408
	s_mul_hi_u32 s39, s4, s35                                  // 000000005A7C: 96A72304
	s_mul_i32 s38, s4, s35                                     // 000000005A80: 96262304
	s_mul_hi_u32 s10, s4, s34                                  // 000000005A84: 968A2204
	s_mul_i32 s31, s5, s34                                     // 000000005A88: 961F2205
	s_add_nc_u64 s[10:11], s[10:11], s[38:39]                  // 000000005A8C: A98A260A
	s_mul_hi_u32 s7, s5, s34                                   // 000000005A90: 96872205
	s_mul_hi_u32 s33, s5, s35                                  // 000000005A94: 96A12305
	s_wait_alu 0xfffe                                          // 000000005A98: BF88FFFE
	s_add_co_u32 s10, s10, s31                                 // 000000005A9C: 800A1F0A
	s_add_co_ci_u32 s36, s11, s7                               // 000000005AA0: 8224070B
	s_mul_i32 s34, s5, s35                                     // 000000005AA4: 96222305
	s_add_co_ci_u32 s35, s33, 0                                // 000000005AA8: 82238021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000005AAC: BF8704C9
	s_add_nc_u64 s[10:11], s[36:37], s[34:35]                  // 000000005AB0: A98A2224
	s_mov_b32 s35, s6                                          // 000000005AB4: BEA30006
	s_add_co_u32 s4, s4, s10                                   // 000000005AB8: 80040A04
	s_cselect_b32 s7, -1, 0                                    // 000000005ABC: 980780C1
	s_cmp_lg_u32 s7, 0                                         // 000000005AC0: BF078007
	s_add_co_ci_u32 s5, s5, s11                                // 000000005AC4: 82050B05
	s_mov_b32 s11, s6                                          // 000000005AC8: BE8B0006
	s_wait_alu 0xfffe                                          // 000000005ACC: BF88FFFE
	s_mul_u64 s[8:9], s[8:9], s[4:5]                           // 000000005AD0: AA880408
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005AD4: BF870009
	s_mul_hi_u32 s37, s4, s9                                   // 000000005AD8: 96A50904
	s_mul_i32 s36, s4, s9                                      // 000000005ADC: 96240904
	s_mul_hi_u32 s10, s4, s8                                   // 000000005AE0: 968A0804
	s_mul_i32 s31, s5, s8                                      // 000000005AE4: 961F0805
	s_add_nc_u64 s[10:11], s[10:11], s[36:37]                  // 000000005AE8: A98A240A
	s_mul_hi_u32 s7, s5, s8                                    // 000000005AEC: 96870805
	s_mul_hi_u32 s33, s5, s9                                   // 000000005AF0: 96A10905
	s_mul_i32 s8, s5, s9                                       // 000000005AF4: 96080905
	s_wait_alu 0xfffe                                          // 000000005AF8: BF88FFFE
	s_add_co_u32 s9, s10, s31                                  // 000000005AFC: 80091F0A
	s_add_co_ci_u32 s34, s11, s7                               // 000000005B00: 8222070B
	s_add_co_ci_u32 s9, s33, 0                                 // 000000005B04: 82098021
	s_mov_b32 s11, s6                                          // 000000005B08: BE8B0006
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]                      // 000000005B0C: A9880822
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005B10: BF870009
	s_add_co_u32 s4, s4, s8                                    // 000000005B14: 80040804
	s_cselect_b32 s7, -1, 0                                    // 000000005B18: 980780C1
	s_wait_alu 0xfffe                                          // 000000005B1C: BF88FFFE
	s_mul_hi_u32 s10, s22, s4                                  // 000000005B20: 968A0416
	s_cmp_lg_u32 s7, 0                                         // 000000005B24: BF078007
	s_mul_hi_u32 s7, s23, s4                                   // 000000005B28: 96870417
	s_add_co_ci_u32 s8, s5, s9                                 // 000000005B2C: 82080905
	s_mul_i32 s9, s23, s4                                      // 000000005B30: 96090417
	s_mul_hi_u32 s5, s22, s8                                   // 000000005B34: 96850816
	s_mul_i32 s4, s22, s8                                      // 000000005B38: 96040816
	s_mul_hi_u32 s31, s23, s8                                  // 000000005B3C: 969F0817
	s_wait_alu 0xfffe                                          // 000000005B40: BF88FFFE
	s_add_nc_u64 s[4:5], s[10:11], s[4:5]                      // 000000005B44: A984040A
	s_mul_i32 s8, s23, s8                                      // 000000005B48: 96080817
	s_wait_alu 0xfffe                                          // 000000005B4C: BF88FFFE
	s_add_co_u32 s4, s4, s9                                    // 000000005B50: 80040904
	s_add_co_ci_u32 s34, s5, s7                                // 000000005B54: 82220705
	s_add_co_ci_u32 s9, s31, 0                                 // 000000005B58: 8209801F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005B5C: BF870009
	s_add_nc_u64 s[4:5], s[34:35], s[8:9]                      // 000000005B60: A9840822
	s_wait_alu 0xfffe                                          // 000000005B64: BF88FFFE
	s_mul_u64 s[8:9], s[24:25], s[4:5]                         // 000000005B68: AA880418
	s_add_nc_u64 s[34:35], s[4:5], 2                           // 000000005B6C: A9A28204
	s_sub_co_u32 s7, s22, s8                                   // 000000005B70: 80870816
	s_cselect_b32 s8, -1, 0                                    // 000000005B74: 980880C1
	s_sub_co_i32 s10, s23, s9                                  // 000000005B78: 818A0917
	s_cmp_lg_u32 s8, 0                                         // 000000005B7C: BF078008
	s_sub_co_ci_u32 s10, s10, s25                              // 000000005B80: 828A190A
	s_sub_co_u32 s11, s7, s24                                  // 000000005B84: 808B1807
	s_cselect_b32 s31, -1, 0                                   // 000000005B88: 981F80C1
	s_wait_alu 0xfffe                                          // 000000005B8C: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000005B90: BF07801F
	s_sub_co_ci_u32 s10, s10, 0                                // 000000005B94: 828A800A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005B98: BF870009
	s_cmp_ge_u32 s10, s25                                      // 000000005B9C: BF09190A
	s_cselect_b32 s31, -1, 0                                   // 000000005BA0: 981F80C1
	s_cmp_ge_u32 s11, s24                                      // 000000005BA4: BF09180B
	s_cselect_b32 s33, -1, 0                                   // 000000005BA8: 982180C1
	s_cmp_eq_u32 s10, s25                                      // 000000005BAC: BF06190A
	s_add_nc_u64 s[10:11], s[4:5], 1                           // 000000005BB0: A98A8104
	s_wait_alu 0xfffe                                          // 000000005BB4: BF88FFFE
	s_cselect_b32 s31, s33, s31                                // 000000005BB8: 981F1F21
	s_wait_alu 0xfffe                                          // 000000005BBC: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000005BC0: BF07801F
	s_cselect_b32 s10, s34, s10                                // 000000005BC4: 980A0A22
	s_cselect_b32 s11, s35, s11                                // 000000005BC8: 980B0B23
	s_cmp_lg_u32 s8, 0                                         // 000000005BCC: BF078008
	s_sub_co_ci_u32 s8, s23, s9                                // 000000005BD0: 82880917
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005BD4: BF870009
	s_cmp_ge_u32 s8, s25                                       // 000000005BD8: BF091908
	s_cselect_b32 s9, -1, 0                                    // 000000005BDC: 980980C1
	s_cmp_ge_u32 s7, s24                                       // 000000005BE0: BF091807
	s_cselect_b32 s7, -1, 0                                    // 000000005BE4: 980780C1
	s_cmp_eq_u32 s8, s25                                       // 000000005BE8: BF061908
	s_cselect_b32 s7, s7, s9                                   // 000000005BEC: 98070907
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005BF0: BF870009
	s_cmp_lg_u32 s7, 0                                         // 000000005BF4: BF078007
	s_cselect_b32 s5, s11, s5                                  // 000000005BF8: 9805050B
	s_cselect_b32 s4, s10, s4                                  // 000000005BFC: 9804040A
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000005C00: 916A067E
	s_cbranch_vccnz 32                                         // 000000005C04: BFA40020 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x388>
	v_cvt_f32_u32_e32 v1, s24                                  // 000000005C08: 7E020C18
	s_sub_co_i32 s5, 0, s24                                    // 000000005C0C: 81851880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005C10: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000005C14: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000005C18: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005C20: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000005C24: 7E020F01
	v_readfirstlane_b32 s4, v1                                 // 000000005C28: 7E080501
	s_wait_alu 0xfffe                                          // 000000005C2C: BF88FFFE
	s_mul_i32 s5, s5, s4                                       // 000000005C30: 96050405
	s_wait_alu 0xfffe                                          // 000000005C34: BF88FFFE
	s_mul_hi_u32 s5, s4, s5                                    // 000000005C38: 96850504
	s_wait_alu 0xfffe                                          // 000000005C3C: BF88FFFE
	s_add_co_i32 s4, s4, s5                                    // 000000005C40: 81040504
	s_wait_alu 0xfffe                                          // 000000005C44: BF88FFFE
	s_mul_hi_u32 s4, s22, s4                                   // 000000005C48: 96840416
	s_wait_alu 0xfffe                                          // 000000005C4C: BF88FFFE
	s_mul_i32 s5, s4, s24                                      // 000000005C50: 96051804
	s_add_co_i32 s6, s4, 1                                     // 000000005C54: 81068104
	s_wait_alu 0xfffe                                          // 000000005C58: BF88FFFE
	s_sub_co_i32 s5, s22, s5                                   // 000000005C5C: 81850516
	s_wait_alu 0xfffe                                          // 000000005C60: BF88FFFE
	s_sub_co_i32 s7, s5, s24                                   // 000000005C64: 81871805
	s_cmp_ge_u32 s5, s24                                       // 000000005C68: BF091805
	s_cselect_b32 s4, s6, s4                                   // 000000005C6C: 98040406
	s_cselect_b32 s5, s7, s5                                   // 000000005C70: 98050507
	s_wait_alu 0xfffe                                          // 000000005C74: BF88FFFE
	s_add_co_i32 s6, s4, 1                                     // 000000005C78: 81068104
	s_cmp_ge_u32 s5, s24                                       // 000000005C7C: BF091805
	s_mov_b32 s5, 0                                            // 000000005C80: BE850080
	s_cselect_b32 s4, s6, s4                                   // 000000005C84: 98040406
	s_mul_u64 s[6:7], s[2:3], s[22:23]                         // 000000005C88: AA861602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000005C8C: BF8704C9
	s_sub_nc_u64 s[6:7], s[28:29], s[6:7]                      // 000000005C90: AA06061C
	s_wait_alu 0xfffe                                          // 000000005C94: BF88FFFE
	s_or_b64 s[8:9], s[6:7], s[4:5]                            // 000000005C98: 8C880406
	s_mov_b32 s8, 0                                            // 000000005C9C: BE880080
	s_cmp_lg_u64 s[8:9], 0                                     // 000000005CA0: BF118008
	s_cbranch_scc0 957                                         // 000000005CA4: BFA103BD <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x129c>
	s_cvt_f32_u32 s9, s4                                       // 000000005CA8: BE896504
	s_cvt_f32_u32 s10, s5                                      // 000000005CAC: BE8A6505
	s_sub_nc_u64 s[22:23], 0, s[4:5]                           // 000000005CB0: AA160480
	s_mov_b32 s35, s8                                          // 000000005CB4: BEA30008
	s_mov_b32 s39, s8                                          // 000000005CB8: BEA70008
	s_fmamk_f32 s9, s10, 0x4f800000, s9                        // 000000005CBC: A309090A 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005CC4: BF87029B
	v_s_rcp_f32 s9, s9                                         // 000000005CC8: D6840009 00000009
	s_mul_f32 s9, s9, 0x5f7ffffc                               // 000000005CD0: A209FF09 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000005CD8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000005CDC: BF87059A
	s_mul_f32 s10, s9, 0x2f800000                              // 000000005CE0: A20AFF09 2F800000
	s_trunc_f32 s10, s10                                       // 000000005CE8: BE8A620A
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000005CEC: BF8704BB
	s_fmamk_f32 s9, s10, 0xcf800000, s9                        // 000000005CF0: A309090A CF800000
	s_cvt_u32_f32 s11, s10                                     // 000000005CF8: BE8B670A
	s_wait_alu 0xfffe                                          // 000000005CFC: BF88FFFE
	s_cvt_u32_f32 s10, s9                                      // 000000005D00: BE8A6709
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000005D04: BF87049B
	s_mul_u64 s[36:37], s[22:23], s[10:11]                     // 000000005D08: AAA40A16
	s_mul_hi_u32 s41, s10, s37                                 // 000000005D0C: 96A9250A
	s_mul_i32 s40, s10, s37                                    // 000000005D10: 9628250A
	s_mul_hi_u32 s34, s10, s36                                 // 000000005D14: 96A2240A
	s_mul_i32 s31, s11, s36                                    // 000000005D18: 961F240B
	s_add_nc_u64 s[34:35], s[34:35], s[40:41]                  // 000000005D1C: A9A22822
	s_mul_hi_u32 s9, s11, s36                                  // 000000005D20: 9689240B
	s_mul_hi_u32 s33, s11, s37                                 // 000000005D24: 96A1250B
	s_wait_alu 0xfffe                                          // 000000005D28: BF88FFFE
	s_add_co_u32 s31, s34, s31                                 // 000000005D2C: 801F1F22
	s_add_co_ci_u32 s38, s35, s9                               // 000000005D30: 82260923
	s_mul_i32 s36, s11, s37                                    // 000000005D34: 9624250B
	s_add_co_ci_u32 s37, s33, 0                                // 000000005D38: 82258021
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005D3C: BF870009
	s_add_nc_u64 s[34:35], s[38:39], s[36:37]                  // 000000005D40: A9A22426
	s_mov_b32 s37, s8                                          // 000000005D44: BEA50008
	s_add_co_u32 s10, s10, s34                                 // 000000005D48: 800A220A
	s_cselect_b32 s9, -1, 0                                    // 000000005D4C: 980980C1
	s_wait_alu 0xfffe                                          // 000000005D50: BF88FFFE
	s_cmp_lg_u32 s9, 0                                         // 000000005D54: BF078009
	s_add_co_ci_u32 s11, s11, s35                              // 000000005D58: 820B230B
	s_mov_b32 s35, s8                                          // 000000005D5C: BEA30008
	s_mul_u64 s[22:23], s[22:23], s[10:11]                     // 000000005D60: AA960A16
	s_wait_alu 0xfffe                                          // 000000005D64: BF88FFFE
	s_mul_hi_u32 s39, s10, s23                                 // 000000005D68: 96A7170A
	s_mul_i32 s38, s10, s23                                    // 000000005D6C: 9626170A
	s_mul_hi_u32 s34, s10, s22                                 // 000000005D70: 96A2160A
	s_mul_i32 s31, s11, s22                                    // 000000005D74: 961F160B
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]                  // 000000005D78: A9A22622
	s_mul_hi_u32 s9, s11, s22                                  // 000000005D7C: 9689160B
	s_mul_hi_u32 s33, s11, s23                                 // 000000005D80: 96A1170B
	s_mul_i32 s22, s11, s23                                    // 000000005D84: 9616170B
	s_wait_alu 0xfffe                                          // 000000005D88: BF88FFFE
	s_add_co_u32 s23, s34, s31                                 // 000000005D8C: 80171F22
	s_add_co_ci_u32 s36, s35, s9                               // 000000005D90: 82240923
	s_add_co_ci_u32 s23, s33, 0                                // 000000005D94: 82178021
	s_mov_b32 s35, s8                                          // 000000005D98: BEA30008
	s_wait_alu 0xfffe                                          // 000000005D9C: BF88FFFE
	s_add_nc_u64 s[22:23], s[36:37], s[22:23]                  // 000000005DA0: A9961624
	s_wait_alu 0xfffe                                          // 000000005DA4: BF88FFFE
	s_add_co_u32 s9, s10, s22                                  // 000000005DA8: 8009160A
	s_cselect_b32 s10, -1, 0                                   // 000000005DAC: 980A80C1
	s_wait_alu 0xfffe                                          // 000000005DB0: BF88FFFE
	s_mul_hi_u32 s34, s6, s9                                   // 000000005DB4: 96A20906
	s_cmp_lg_u32 s10, 0                                        // 000000005DB8: BF07800A
	s_mul_hi_u32 s31, s7, s9                                   // 000000005DBC: 969F0907
	s_add_co_ci_u32 s22, s11, s23                              // 000000005DC0: 8216170B
	s_mul_i32 s9, s7, s9                                       // 000000005DC4: 96090907
	s_wait_alu 0xfffe                                          // 000000005DC8: BF88FFFE
	s_mul_hi_u32 s11, s6, s22                                  // 000000005DCC: 968B1606
	s_mul_i32 s10, s6, s22                                     // 000000005DD0: 960A1606
	s_mul_hi_u32 s23, s7, s22                                  // 000000005DD4: 96971607
	s_add_nc_u64 s[10:11], s[34:35], s[10:11]                  // 000000005DD8: A98A0A22
	s_mul_i32 s22, s7, s22                                     // 000000005DDC: 96161607
	s_add_co_u32 s9, s10, s9                                   // 000000005DE0: 8009090A
	s_add_co_ci_u32 s36, s11, s31                              // 000000005DE4: 82241F0B
	s_wait_alu 0xfffe                                          // 000000005DE8: BF88FFFE
	s_add_co_ci_u32 s23, s23, 0                                // 000000005DEC: 82178017
	s_wait_alu 0xfffe                                          // 000000005DF0: BF88FFFE
	s_add_nc_u64 s[10:11], s[36:37], s[22:23]                  // 000000005DF4: A98A1624
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005DF8: BF870009
	s_mul_u64 s[22:23], s[4:5], s[10:11]                       // 000000005DFC: AA960A04
	s_wait_alu 0xfffe                                          // 000000005E00: BF88FFFE
	s_sub_co_u32 s9, s6, s22                                   // 000000005E04: 80891606
	s_cselect_b32 s22, -1, 0                                   // 000000005E08: 981680C1
	s_sub_co_i32 s31, s7, s23                                  // 000000005E0C: 819F1707
	s_wait_alu 0xfffe                                          // 000000005E10: BF88FFFE
	s_cmp_lg_u32 s22, 0                                        // 000000005E14: BF078016
	s_sub_co_ci_u32 s31, s31, s5                               // 000000005E18: 829F051F
	s_sub_co_u32 s33, s9, s4                                   // 000000005E1C: 80A10409
	s_cselect_b32 s34, -1, 0                                   // 000000005E20: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005E24: BF870009
	s_cmp_lg_u32 s34, 0                                        // 000000005E28: BF078022
	s_add_nc_u64 s[34:35], s[10:11], 1                         // 000000005E2C: A9A2810A
	s_wait_alu 0xfffe                                          // 000000005E30: BF88FFFE
	s_sub_co_ci_u32 s31, s31, 0                                // 000000005E34: 829F801F
	s_wait_alu 0xfffe                                          // 000000005E38: BF88FFFE
	s_cmp_ge_u32 s31, s5                                       // 000000005E3C: BF09051F
	s_cselect_b32 s36, -1, 0                                   // 000000005E40: 982480C1
	s_cmp_ge_u32 s33, s4                                       // 000000005E44: BF090421
	s_cselect_b32 s33, -1, 0                                   // 000000005E48: 982180C1
	s_cmp_eq_u32 s31, s5                                       // 000000005E4C: BF06051F
	s_cselect_b32 s31, s33, s36                                // 000000005E50: 981F2421
	s_add_nc_u64 s[36:37], s[10:11], 2                         // 000000005E54: A9A4820A
	s_wait_alu 0xfffe                                          // 000000005E58: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000005E5C: BF07801F
	s_cselect_b32 s31, s36, s34                                // 000000005E60: 981F2224
	s_cselect_b32 s33, s37, s35                                // 000000005E64: 98212325
	s_cmp_lg_u32 s22, 0                                        // 000000005E68: BF078016
	s_sub_co_ci_u32 s7, s7, s23                                // 000000005E6C: 82871707
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005E70: BF870009
	s_cmp_ge_u32 s7, s5                                        // 000000005E74: BF090507
	s_cselect_b32 s22, -1, 0                                   // 000000005E78: 981680C1
	s_cmp_ge_u32 s9, s4                                        // 000000005E7C: BF090409
	s_cselect_b32 s9, -1, 0                                    // 000000005E80: 980980C1
	s_cmp_eq_u32 s7, s5                                        // 000000005E84: BF060507
	s_wait_alu 0xfffe                                          // 000000005E88: BF88FFFE
	s_cselect_b32 s5, s9, s22                                  // 000000005E8C: 98051609
	s_wait_alu 0xfffe                                          // 000000005E90: BF88FFFE
	s_cmp_lg_u32 s5, 0                                         // 000000005E94: BF078005
	s_cselect_b32 s23, s33, s11                                // 000000005E98: 98170B21
	s_cselect_b32 s22, s31, s10                                // 000000005E9C: 98160A1F
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000005EA0: 916A087E
	s_cbranch_vccnz 29                                         // 000000005EA4: BFA4001D <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x61c>
	v_cvt_f32_u32_e32 v1, s4                                   // 000000005EA8: 7E020C04
	s_sub_co_i32 s7, 0, s4                                     // 000000005EAC: 81870480
	s_mov_b32 s23, 0                                           // 000000005EB0: BE970080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005EB4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000005EB8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000005EBC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005EC4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000005EC8: 7E020F01
	v_readfirstlane_b32 s5, v1                                 // 000000005ECC: 7E0A0501
	s_mul_i32 s7, s7, s5                                       // 000000005ED0: 96070507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000005ED4: BF870499
	s_mul_hi_u32 s7, s5, s7                                    // 000000005ED8: 96870705
	s_add_co_i32 s5, s5, s7                                    // 000000005EDC: 81050705
	s_wait_alu 0xfffe                                          // 000000005EE0: BF88FFFE
	s_mul_hi_u32 s5, s6, s5                                    // 000000005EE4: 96850506
	s_wait_alu 0xfffe                                          // 000000005EE8: BF88FFFE
	s_mul_i32 s7, s5, s4                                       // 000000005EEC: 96070405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005EF0: BF870009
	s_sub_co_i32 s6, s6, s7                                    // 000000005EF4: 81860706
	s_add_co_i32 s7, s5, 1                                     // 000000005EF8: 81078105
	s_sub_co_i32 s8, s6, s4                                    // 000000005EFC: 81880406
	s_cmp_ge_u32 s6, s4                                        // 000000005F00: BF090406
	s_cselect_b32 s5, s7, s5                                   // 000000005F04: 98050507
	s_wait_alu 0xfffe                                          // 000000005F08: BF88FFFE
	s_cselect_b32 s6, s8, s6                                   // 000000005F0C: 98060608
	s_add_co_i32 s7, s5, 1                                     // 000000005F10: 81078105
	s_cmp_ge_u32 s6, s4                                        // 000000005F14: BF090406
	s_cselect_b32 s22, s7, s5                                  // 000000005F18: 98160507
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v8, 2, v0       // 000000005F1C: CA220080 01080082
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]                      // 000000005F24: A9820212
	s_mov_b64 s[34:35], 0                                      // 000000005F28: BEA20180
	s_wait_alu 0xfffe                                          // 000000005F2C: BF88FFFE
	s_add_nc_u64 s[18:19], s[2:3], 1                           // 000000005F30: A9928102
	v_cmp_gt_u64_e64 s2, s[20:21], v[0:1]                      // 000000005F34: D45C0002 00020014
	v_cmp_le_u64_e64 s3, s[20:21], v[0:1]                      // 000000005F3C: D45B0003 00020014
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v11, v1            // 000000005F44: CA100101 040A0101
	s_cmp_eq_u64 s[18:19], 0                                   // 000000005F4C: BF108012
	s_cbranch_scc1 748                                         // 000000005F50: BFA202EC <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1204>
	v_mbcnt_lo_u32_b32 v3, -1, 0                               // 000000005F54: D71F0003 000100C1
	v_dual_mov_b32 v11, 0 :: v_dual_and_b32 v2, 31, v0         // 000000005F5C: CA240080 0B02009F
	s_load_b32 s31, s[0:1], 0x48                               // 000000005F64: F40007C0 F8000048
	s_cmp_gt_u32 s30, 1                                        // 000000005F6C: BF08811E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005F70: BF870112
	v_xor_b32_e32 v4, 31, v3                                   // 000000005F74: 3A08069F
	v_cmp_eq_u32_e64 s5, 0, v2                                 // 000000005F78: D44A0005 00020480
	v_cmp_gt_u32_e64 s7, 8, v2                                 // 000000005F80: D44C0007 00020488
	v_lshlrev_b32_e32 v10, 2, v2                               // 000000005F88: 30140482
	s_mul_u64 s[38:39], s[26:27], s[28:29]                     // 000000005F8C: AAA61C1A
	v_cmp_gt_u32_e32 vcc_lo, 8, v4                             // 000000005F90: 7C980888
	v_and_b32_e32 v2, 16, v4                                   // 000000005F94: 36040890
	s_cselect_b32 s33, -1, 0                                   // 000000005F98: 982180C1
	s_lshl_b64 s[38:39], s[38:39], 2                           // 000000005F9C: 84A68226
	v_dual_mov_b32 v18, 0 :: v_dual_lshlrev_b32 v17, 2, v0     // 000000005FA0: CA220080 12100082
	v_cndmask_b32_e64 v5, 8, 0, vcc_lo                         // 000000005FA8: D5010005 01A90088
	v_cmp_gt_u32_e32 vcc_lo, 4, v4                             // 000000005FB0: 7C980884
	v_add_co_u32 v19, s9, s16, v8                              // 000000005FB4: D7000913 00021010
	s_add_nc_u64 s[12:13], s[12:13], s[38:39]                  // 000000005FBC: A98C260C
	v_add_lshl_u32 v12, v2, v3, 2                              // 000000005FC0: D647000C 020A0702
	s_wait_alu 0xfffd                                          // 000000005FC8: BF88FFFD
	v_cndmask_b32_e64 v6, 4, 0, vcc_lo                         // 000000005FCC: D5010006 01A90084
	v_cmp_gt_u32_e32 vcc_lo, 2, v4                             // 000000005FD4: 7C980882
	s_wait_alu 0xf1ff                                          // 000000005FD8: BF88F1FF
	v_add_co_ci_u32_e64 v20, null, s17, 0, s9                  // 000000005FDC: D5207C14 00250011
	v_add_co_u32 v2, s9, s12, v8                               // 000000005FE4: D7000902 0002100C
	s_wait_alu 0xfffd                                          // 000000005FEC: BF88FFFD
	v_cndmask_b32_e64 v4, 2, 0, vcc_lo                         // 000000005FF0: D5010004 01A90082
	v_cmp_ne_u32_e32 vcc_lo, 31, v3                            // 000000005FF8: 7C9A069F
	v_cmp_gt_u64_e64 s4, s[26:27], v[0:1]                      // 000000005FFC: D45C0004 0002001A
	v_lshrrev_b32_e32 v9, 3, v0                                // 000000006004: 32120083
	v_cmp_gt_u32_e64 s6, 32, v0                                // 000000006008: D44C0006 000200A0
	v_cmp_eq_u32_e64 s8, 0, v0                                 // 000000006010: D44A0008 00020080
	s_wait_alu 0xfffd                                          // 000000006018: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo                // 00000000601C: D5207C07 01AA0680
	v_add_lshl_u32 v13, v5, v3, 2                              // 000000006024: D647000D 020A0705
	v_add_lshl_u32 v14, v6, v3, 2                              // 00000000602C: D647000E 020A0706
	v_add_lshl_u32 v15, v4, v3, 2                              // 000000006034: D647000F 020A0704
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000603C: BF870004
	v_lshlrev_b32_e32 v16, 2, v7                               // 000000006040: 30200E82
	s_wait_alu 0xf1ff                                          // 000000006044: BF88F1FF
	v_add_co_ci_u32_e64 v3, null, s13, 0, s9                   // 000000006048: D5207C03 0025000D
	v_dual_mov_b32 v22, 0 :: v_dual_add_nc_u32 v21, 0x400, v17 // 000000006050: CA200080 161422FF 00000400
	s_mov_b32 s11, 0                                           // 00000000605C: BE8B0080
	s_lshr_b32 s36, s30, 1                                     // 000000006060: 8524811E
	s_lshl_b32 s37, s30, 2                                     // 000000006064: 8425821E
	s_lshl_b32 s38, s30, 2                                     // 000000006068: 8426821E
	s_mov_b32 s39, 0xff7fffff                                  // 00000000606C: BEA700FF FF7FFFFF
	s_sub_nc_u64 s[12:13], s[18:19], s[34:35]                  // 000000006074: AA0C2212
	s_mov_b32 s10, s11                                         // 000000006078: BE8A000B
	s_wait_alu 0xfffe                                          // 00000000607C: BF88FFFE
	v_cmp_lt_u64_e64 s9, s[12:13], 64                          // 000000006080: D4590009 0001800C
	s_and_b32 s9, s9, exec_lo                                  // 000000006088: 8B097E09
	s_cselect_b32 s40, s12, 64                                 // 00000000608C: 9828C00C
	s_branch 12                                                // 000000006090: BFA0000C <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x7c4>
	s_wait_alu 0xfffe                                          // 000000006094: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000006098: 8C7E097E
	s_add_co_i32 s10, s10, 1                                   // 00000000609C: 810A810A
	s_wait_loadcnt_dscnt 0x0                                   // 0000000060A0: BFC80000
	s_wait_alu 0xfffe                                          // 0000000060A4: BF88FFFE
	s_cmp_ge_u32 s10, s40                                      // 0000000060A8: BF09280A
	s_barrier_signal -1                                        // 0000000060AC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000060B0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000060B4: EE0AC07C 00040000 00000000
	s_cbranch_scc1 141                                         // 0000000060C0: BFA2008D <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x9f8>
	v_mov_b32_e32 v23, 0                                       // 0000000060C4: 7E2E0280
	s_and_saveexec_b32 s41, s4                                 // 0000000060C8: BEA92004
	s_cbranch_execz 50                                         // 0000000060CC: BFA50032 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x898>
	s_add_nc_u64 s[16:17], s[34:35], s[10:11]                  // 0000000060D0: A9900A22
	v_mov_b32_e32 v5, v3                                       // 0000000060D4: 7E0A0303
	s_wait_alu 0xfffe                                          // 0000000060D8: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[24:25]                     // 0000000060DC: AA901810
	v_mov_b32_e32 v7, v1                                       // 0000000060E0: 7E0E0301
	s_wait_alu 0xfffe                                          // 0000000060E4: BF88FFFE
	s_add_nc_u64 s[16:17], s[16:17], s[22:23]                  // 0000000060E8: A9901610
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v4, v2             // 0000000060EC: CA100080 17040102
	s_wait_alu 0xfffe                                          // 0000000060F4: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[26:27]                     // 0000000060F8: AA901A10
	v_mov_b32_e32 v6, v0                                       // 0000000060FC: 7E0C0300
	s_wait_alu 0xfffe                                          // 000000006100: BF88FFFE
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000006104: 84908210
	s_mov_b32 s42, 0                                           // 000000006108: BEAA0080
	s_wait_alu 0xfffe                                          // 00000000610C: BF88FFFE
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]                  // 000000006110: A990100E
	v_lshlrev_b64_e32 v[24:25], 2, v[6:7]                      // 000000006114: 3E300C82
	s_wait_alu 0xfffe                                          // 000000006118: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000611C: BF870121
	v_add_co_u32 v24, vcc_lo, s16, v24                         // 000000006120: D7006A18 00023010
	s_wait_alu 0xfffd                                          // 000000006128: BF88FFFD
	v_add_co_ci_u32_e64 v25, null, s17, v25, vcc_lo            // 00000000612C: D5207C19 01AA3211
	v_add_co_u32 v6, vcc_lo, v6, s30                           // 000000006134: D7006A06 00003D06
	global_load_b32 v26, v[4:5], off                           // 00000000613C: EE05007C 0000001A 00000004
	global_load_b32 v24, v[24:25], off                         // 000000006148: EE05007C 00000018 00000018
	s_wait_alu 0xfffd                                          // 000000006154: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo                // 000000006158: D5207C07 01AA0E80
	v_add_co_u32 v4, s9, v4, s37                               // 000000006160: D7000904 00004B04
	s_wait_alu 0xf1ff                                          // 000000006168: BF88F1FF
	v_add_co_ci_u32_e64 v5, null, 0, v5, s9                    // 00000000616C: D5207C05 00260A80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006174: BF870003
	v_cmp_le_u64_e32 vcc_lo, s[26:27], v[6:7]                  // 000000006178: 7CB60C1A
	s_or_b32 s42, vcc_lo, s42                                  // 00000000617C: 8C2A2A6A
	s_wait_loadcnt 0x0                                         // 000000006180: BFC00000
	v_fmac_f32_e32 v23, v26, v24                               // 000000006184: 562E311A
	s_wait_alu 0xfffe                                          // 000000006188: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s42                       // 00000000618C: 917E2A7E
	s_cbranch_execnz 65504                                     // 000000006190: BFA6FFE0 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x814>
	s_or_b32 exec_lo, exec_lo, s42                             // 000000006194: 8C7E2A7E
	s_wait_alu 0xfffe                                          // 000000006198: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s41                             // 00000000619C: 8C7E297E
	ds_bpermute_b32 v4, v12, v23                               // 0000000061A0: DACC0000 0400170C
	s_wait_dscnt 0x0                                           // 0000000061A8: BFC60000
	v_add_f32_e32 v4, v23, v4                                  // 0000000061AC: 06080917
	ds_bpermute_b32 v5, v13, v4                                // 0000000061B0: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 0000000061B8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000061BC: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 0000000061C0: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000061C8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000061CC: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 0000000061D0: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000061D8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000061DC: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 0000000061E0: DACC0000 05000410
	s_and_saveexec_b32 s9, s5                                  // 0000000061E8: BE892005
	s_cbranch_execz 4                                          // 0000000061EC: BFA50004 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x900>
	s_wait_dscnt 0x0                                           // 0000000061F0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000061F4: 06080B04
	ds_store_b32 v9, v4                                        // 0000000061F8: D8340000 00000409
	s_wait_alu 0xfffe                                          // 000000006200: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000006204: 8C7E097E
	s_wait_dscnt 0x0                                           // 000000006208: BFC60000
	s_barrier_signal -1                                        // 00000000620C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006210: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000006214: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s6                                  // 000000006220: BE892006
	s_cbranch_execz 31                                         // 000000006224: BFA5001F <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x9a4>
	v_mov_b32_e32 v4, 0                                        // 000000006228: 7E080280
	s_and_saveexec_b32 s16, s7                                 // 00000000622C: BE902007
	ds_load_b32 v4, v10                                        // 000000006230: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000006238: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 00000000623C: 8C7E107E
	s_wait_dscnt 0x0                                           // 000000006240: BFC60000
	ds_bpermute_b32 v5, v12, v4                                // 000000006244: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 00000000624C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000006250: 06080B04
	ds_bpermute_b32 v5, v13, v4                                // 000000006254: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 00000000625C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000006260: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 000000006264: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 00000000626C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000006270: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 000000006274: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 00000000627C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000006280: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 000000006284: DACC0000 05000410
	s_and_b32 exec_lo, exec_lo, s5                             // 00000000628C: 8B7E057E
	s_cbranch_execz 4                                          // 000000006290: BFA50004 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x9a4>
	s_wait_dscnt 0x0                                           // 000000006294: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000006298: 06080B04
	ds_store_b32 v18, v4                                       // 00000000629C: D8340000 00000412
	s_wait_alu 0xfffe                                          // 0000000062A4: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000062A8: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000062AC: BFC80000
	s_barrier_signal -1                                        // 0000000062B0: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000062B4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000062B8: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s8                                  // 0000000062C4: BE892008
	s_cbranch_execz 65394                                      // 0000000062C8: BFA5FF72 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x794>
	ds_load_b32 v4, v18                                        // 0000000062CC: D8D80000 04000012
	s_lshl_b32 s16, s10, 2                                     // 0000000062D4: 8410820A
	s_wait_dscnt 0x0                                           // 0000000062D8: BFC60000
	s_wait_kmcnt 0x0                                           // 0000000062DC: BFC70000
	s_wait_alu 0xfffe                                          // 0000000062E0: BF88FFFE
	v_dual_mov_b32 v5, s16 :: v_dual_mul_f32 v4, s31, v4       // 0000000062E4: CA060010 0504081F
	ds_store_b32 v5, v4 offset:1024                            // 0000000062EC: D8340400 00000405
	s_branch 65383                                             // 0000000062F4: BFA0FF67 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x794>
	v_cmp_gt_u32_e32 vcc_lo, s40, v0                           // 0000000062F8: 7C980028
	v_mov_b32_e32 v4, 0xff7fffff                               // 0000000062FC: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s16, vcc_lo                             // 000000006304: BE90206A
	s_cbranch_execz 25                                         // 000000006308: BFA50019 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xa70>
	v_dual_mov_b32 v4, 0xff7fffff :: v_dual_mov_b32 v5, v21    // 00000000630C: CA1000FF 04040115 FF7FFFFF
	v_mov_b32_e32 v6, v0                                       // 000000006318: 7E0C0300
	s_mov_b32 s17, 0                                           // 00000000631C: BE910080
	ds_load_b32 v7, v5                                         // 000000006320: D8D80000 07000005
	v_add_nc_u32_e32 v6, s30, v6                               // 000000006328: 4A0C0C1E
	v_add_nc_u32_e32 v5, s38, v5                               // 00000000632C: 4A0A0A26
	s_delay_alu instid0(VALU_DEP_2)                            // 000000006330: BF870002
	v_cmp_le_u32_e64 s9, s40, v6                               // 000000006334: D44B0009 00020C28
	s_wait_alu 0xfffe                                          // 00000000633C: BF88FFFE
	s_or_b32 s17, s9, s17                                      // 000000006340: 8C111109
	s_wait_dscnt 0x0                                           // 000000006344: BFC60000
	v_cmp_gt_f32_e64 s10, v7, v4                               // 000000006348: D414000A 00020907
	s_wait_alu 0xf1ff                                          // 000000006350: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006354: BF870001
	v_cndmask_b32_e64 v4, v4, v7, s10                          // 000000006358: D5010004 002A0F04
	s_wait_alu 0xfffe                                          // 000000006360: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s17                       // 000000006364: 917E117E
	s_cbranch_execnz 65517                                     // 000000006368: BFA6FFED <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xa20>
	s_or_b32 exec_lo, exec_lo, s17                             // 00000000636C: 8C7E117E
	s_wait_alu 0xfffe                                          // 000000006370: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 000000006374: 8C7E107E
	ds_bpermute_b32 v5, v12, v4                                // 000000006378: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000006380: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000006384: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 00000000638C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000006390: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000006394: D5010004 00260905
	ds_bpermute_b32 v5, v13, v4                                // 00000000639C: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 0000000063A4: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000063A8: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000063B0: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000063B4: D5010004 00260905
	ds_bpermute_b32 v5, v14, v4                                // 0000000063BC: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000063C4: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000063C8: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000063D0: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000063D4: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000063D8: D5010004 00260905
	ds_bpermute_b32 v5, v15, v4                                // 0000000063E0: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000063E8: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000063EC: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000063F4: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000063F8: D5010004 00260905
	ds_bpermute_b32 v5, v16, v4                                // 000000006400: DACC0000 05000410
	s_and_saveexec_b32 s10, s5                                 // 000000006408: BE8A2005
	s_cbranch_execz 9                                          // 00000000640C: BFA50009 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xb34>
	s_wait_dscnt 0x0                                           // 000000006410: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000006414: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 00000000641C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006420: BF870001
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000006424: D5010004 00260905
	ds_store_b32 v9, v4                                        // 00000000642C: D8340000 00000409
	s_wait_alu 0xfffe                                          // 000000006434: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000006438: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 00000000643C: BFC80000
	s_barrier_signal -1                                        // 000000006440: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006444: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000006448: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s10, s6                                 // 000000006454: BE8A2006
	s_cbranch_execz 55                                         // 000000006458: BFA50037 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xc38>
	v_mov_b32_e32 v4, 0xff7fffff                               // 00000000645C: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s9, s7                                  // 000000006464: BE892007
	ds_load_b32 v4, v10                                        // 000000006468: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000006470: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000006474: 8C7E097E
	s_wait_dscnt 0x0                                           // 000000006478: BFC60000
	ds_bpermute_b32 v5, v12, v4                                // 00000000647C: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000006484: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000006488: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000006490: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000006494: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000006498: D5010004 00260905
	ds_bpermute_b32 v5, v13, v4                                // 0000000064A0: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 0000000064A8: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000064AC: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000064B4: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000064B8: D5010004 00260905
	ds_bpermute_b32 v5, v14, v4                                // 0000000064C0: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000064C8: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000064CC: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000064D4: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000064D8: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000064DC: D5010004 00260905
	ds_bpermute_b32 v5, v15, v4                                // 0000000064E4: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000064EC: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000064F0: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000064F8: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000064FC: D5010004 00260905
	ds_bpermute_b32 v5, v16, v4                                // 000000006504: DACC0000 05000410
	s_and_b32 exec_lo, exec_lo, s5                             // 00000000650C: 8B7E057E
	s_cbranch_execz 9                                          // 000000006510: BFA50009 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xc38>
	s_wait_dscnt 0x0                                           // 000000006514: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000006518: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000006520: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006524: BF870001
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000006528: D5010004 00260905
	ds_store_b32 v18, v4                                       // 000000006530: D8340000 00000412
	s_wait_alu 0xfffe                                          // 000000006538: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 00000000653C: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000006540: BFC80000
	s_barrier_signal -1                                        // 000000006544: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006548: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 00000000654C: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s10, s8                                 // 000000006558: BE8A2008
	s_cbranch_execz 57                                         // 00000000655C: BFA50039 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xd44>
	ds_load_b32 v6, v18                                        // 000000006560: D8D80000 06000012
	s_wait_dscnt 0x0                                           // 000000006568: BFC60000
	v_readfirstlane_b32 s9, v6                                 // 00000000656C: 7E120506
	s_cmp_gt_f32 s39, s9                                       // 000000006570: BF440927
	s_cselect_b32 s16, s39, s9                                 // 000000006574: 98100927
	s_wait_alu 0xfffe                                          // 000000006578: BF88FFFE
	s_sub_f32 s17, s39, s16                                    // 00000000657C: A0911027
	v_mov_b32_e32 v5, s16                                      // 000000006580: 7E0A0210
	s_wait_alu 0xfffe                                          // 000000006584: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000006588: BF870529
	s_mul_f32 s9, s17, 0x3fb8aa3b                              // 00000000658C: A209FF11 3FB8AA3B
	s_wait_alu 0xfffe                                          // 000000006594: BF88FFFE
	s_xor_b32 s41, s9, 0x80000000                              // 000000006598: 8D29FF09 80000000
	s_rndne_f32 s42, s9                                        // 0000000065A0: BEAA6309
	s_wait_alu 0xfffe                                          // 0000000065A4: BF88FFFE
	s_fmamk_f32 s41, s17, 0x3fb8aa3b, s41                      // 0000000065A8: A3292911 3FB8AA3B
	s_cmp_nlt_f32 s17, 0xc2ce8ed0                              // 0000000065B0: BF4EFF11 C2CE8ED0
	s_sub_f32 s9, s9, s42                                      // 0000000065B8: A0892A09
	s_wait_alu 0xfffe                                          // 0000000065BC: BF88FFFE
	s_fmamk_f32 s41, s17, 0x32a5705f, s41                      // 0000000065C0: A3292911 32A5705F
	s_wait_alu 0xfffe                                          // 0000000065C8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000065CC: BF8704BA
	s_add_f32 s9, s9, s41                                      // 0000000065D0: A0092909
	s_cvt_i32_f32 s41, s42                                     // 0000000065D4: BEA9662A
	s_wait_alu 0xfffe                                          // 0000000065D8: BF88FFFE
	v_s_exp_f32 s9, s9                                         // 0000000065DC: D6800009 00000009
	s_wait_alu 0xf1ff                                          // 0000000065E4: BF88F1FF
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000065E8: BF8700C5
	v_ldexp_f32 v4, s9, s41                                    // 0000000065EC: D71C0004 00005209
	s_cselect_b32 s9, -1, 0                                    // 0000000065F4: 980980C1
	s_cmp_ngt_f32 s17, 0x42b17218                              // 0000000065F8: BF4BFF11 42B17218
	s_wait_alu 0xfffe                                          // 000000006600: BF88FFFE
	v_cndmask_b32_e64 v4, 0, v4, s9                            // 000000006604: D5010004 00260880
	s_cselect_b32 s9, -1, 0                                    // 00000000660C: 980980C1
	s_cmp_nle_f32 s39, 0xff61b1e6                              // 000000006610: BF4CFF27 FF61B1E6
	s_wait_alu 0xfffe                                          // 000000006618: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 00000000661C: BF8700B1
	v_cndmask_b32_e64 v4, 0x7f800000, v4, s9                   // 000000006620: D5010004 002608FF 7F800000
	s_cselect_b32 s9, -1, 0                                    // 00000000662C: 980980C1
	s_wait_alu 0xfffe                                          // 000000006630: BF88FFFE
	v_cndmask_b32_e64 v4, 0, v4, s9                            // 000000006634: D5010004 00260880
	ds_store_b96 v18, v[4:6] offset:1280                       // 00000000663C: DB780500 00000412
	s_wait_alu 0xfffe                                          // 000000006644: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000006648: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 00000000664C: BFC80000
	s_barrier_signal -1                                        // 000000006650: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006654: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000006658: EE0AC07C 00040000 00000000
	ds_load_b32 v4, v18 offset:1284                            // 000000006664: D8D80504 04000012
	s_wait_dscnt 0x0                                           // 00000000666C: BFC60000
	v_readfirstlane_b32 s39, v4                                // 000000006670: 7E4E0504
	v_mov_b32_e32 v4, 0                                        // 000000006674: 7E080280
	s_and_saveexec_b32 s9, vcc_lo                              // 000000006678: BE89206A
	s_cbranch_execz 48                                         // 00000000667C: BFA50030 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xe40>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, v21             // 000000006680: CA100080 04040115
	v_mov_b32_e32 v6, v0                                       // 000000006688: 7E0C0300
	s_mov_b32 s10, 0                                           // 00000000668C: BE8A0080
	ds_load_b32 v7, v5                                         // 000000006690: D8D80000 07000005
	s_wait_dscnt 0x0                                           // 000000006698: BFC60000
	s_wait_alu 0xf1ff                                          // 00000000669C: BF88F1FF
	v_dual_subrev_f32 v7, s39, v7 :: v_dual_add_nc_u32 v6, s30, v6// 0000000066A0: C9A00E27 07060C1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000066A8: BF870121
	v_mul_f32_e32 v23, 0x3fb8aa3b, v7                          // 0000000066AC: 102E0EFF 3FB8AA3B
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v7                   // 0000000066B4: 7C360EFF C2CE8ED0
	v_fma_f32 v24, 0x3fb8aa3b, v7, -v23                        // 0000000066BC: D6130018 845E0EFF 3FB8AA3B
	v_rndne_f32_e32 v25, v23                                   // 0000000066C8: 7E324717
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000066CC: BF870091
	v_dual_fmac_f32 v24, 0x32a5705f, v7 :: v_dual_sub_f32 v23, v23, v25// 0000000066D0: C80A0EFF 18163317 32A5705F
	v_add_f32_e32 v23, v23, v24                                // 0000000066DC: 062E3117
	v_cvt_i32_f32_e32 v24, v25                                 // 0000000066E0: 7E301119
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000066E4: BF870292
	v_exp_f32_e32 v23, v23                                     // 0000000066E8: 7E2E4B17
	v_ldexp_f32 v23, v23, v24                                  // 0000000066EC: D71C0017 00023117
	s_wait_alu 0xfffd                                          // 0000000066F4: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000066F8: BF870131
	v_cndmask_b32_e32 v23, 0, v23, vcc_lo                      // 0000000066FC: 022E2E80
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v7                   // 000000006700: 7C3C0EFF 42B17218
	s_wait_alu 0xfffd                                          // 000000006708: BF88FFFD
	v_cndmask_b32_e32 v7, 0x7f800000, v23, vcc_lo              // 00000000670C: 020E2EFF 7F800000
	v_cmp_le_u32_e32 vcc_lo, s40, v6                           // 000000006714: 7C960C28
	ds_store_b32 v5, v7                                        // 000000006718: D8340000 00000705
	v_dual_add_f32 v4, v4, v7 :: v_dual_add_nc_u32 v5, s38, v5 // 000000006720: C9200F04 04040A26
	s_wait_alu 0xfffe                                          // 000000006728: BF88FFFE
	s_or_b32 s10, vcc_lo, s10                                  // 00000000672C: 8C0A0A6A
	s_wait_alu 0xfffe                                          // 000000006730: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s10                       // 000000006734: 917E0A7E
	s_cbranch_execnz 65493                                     // 000000006738: BFA6FFD5 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xd90>
	s_or_b32 exec_lo, exec_lo, s10                             // 00000000673C: 8C7E0A7E
	s_wait_alu 0xfffe                                          // 000000006740: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000006744: 8C7E097E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006748: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 00000000674C: 916A217E
	s_mov_b32 s9, s36                                          // 000000006750: BE890024
	ds_store_b32 v17, v4                                       // 000000006754: D8340000 00000411
	s_wait_loadcnt_dscnt 0x0                                   // 00000000675C: BFC80000
	s_barrier_signal -1                                        // 000000006760: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006764: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000006768: EE0AC07C 00040000 00000000
	s_wait_alu 0xfffe                                          // 000000006774: BF88FFFE
	s_cbranch_vccz 142                                         // 000000006778: BFA3008E <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x10b4>
	s_and_saveexec_b32 s9, s8                                  // 00000000677C: BE892008
	s_cbranch_execz 5                                          // 000000006780: BFA50005 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xe98>
	ds_load_b32 v4, v18                                        // 000000006784: D8D80000 04000012
	s_wait_dscnt 0x0                                           // 00000000678C: BFC60000
	ds_store_b32 v18, v4 offset:1292                           // 000000006790: D834050C 00000412
	s_wait_alu 0xfffe                                          // 000000006798: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000679C: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000067A0: BFC80000
	s_barrier_signal -1                                        // 0000000067A4: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000067A8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000067AC: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s3                                  // 0000000067B8: BE892003
	s_wait_alu 0xfffe                                          // 0000000067BC: BF88FFFE
	s_xor_b32 s9, exec_lo, s9                                  // 0000000067C0: 8D09097E
	ds_load_b32 v5, v18 offset:1280                            // 0000000067C4: D8D80500 05000012
	s_wait_alu 0xfffe                                          // 0000000067CC: BF88FFFE
	s_and_not1_saveexec_b32 s9, s9                             // 0000000067D0: BE893009
	s_cbranch_execz 182                                        // 0000000067D4: BFA500B6 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x11b0>
	v_cmp_lt_u64_e64 s10, s[12:13], 4                          // 0000000067D8: D459000A 0001080C
	v_mov_b32_e32 v4, 0                                        // 0000000067E0: 7E080280
	s_max_u32 s13, s40, 1                                      // 0000000067E4: 8A8D8128
	s_and_b32 vcc_lo, exec_lo, s10                             // 0000000067E8: 8B6A0A7E
	s_wait_alu 0xfffe                                          // 0000000067EC: BF88FFFE
	s_cbranch_vccnz 128                                        // 0000000067F0: BFA40080 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x10f4>
	s_and_b32 s12, s13, 0x7c                                   // 0000000067F4: 8B0CFF0D 0000007C
	s_mov_b32 s10, 0                                           // 0000000067FC: BE8A0080
	s_movk_i32 s16, 0x400                                      // 000000006800: B0100400
	s_wait_alu 0xfffe                                          // 000000006804: BF88FFFE
	s_add_nc_u64 s[40:41], s[34:35], s[10:11]                  // 000000006808: A9A80A22
	s_or_b32 s42, s10, 1                                       // 00000000680C: 8C2A810A
	s_mov_b32 s43, s11                                         // 000000006810: BEAB000B
	s_wait_alu 0xfffe                                          // 000000006814: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[24:25]                     // 000000006818: AAA81828
	s_add_nc_u64 s[42:43], s[34:35], s[42:43]                  // 00000000681C: A9AA2A22
	s_wait_alu 0xfffe                                          // 000000006820: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[22:23]                  // 000000006824: A9A81628
	s_mul_u64 s[42:43], s[42:43], s[24:25]                     // 000000006828: AAAA182A
	s_wait_alu 0xfffe                                          // 00000000682C: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 000000006830: AAA81428
	s_add_nc_u64 s[42:43], s[42:43], s[22:23]                  // 000000006834: A9AA162A
	s_wait_alu 0xfffe                                          // 000000006838: BF88FFFE
	s_lshl_b64 s[40:41], s[40:41], 2                           // 00000000683C: 84A88228
	s_or_b32 s44, s10, 2                                       // 000000006840: 8C2C820A
	s_mov_b32 s45, s11                                         // 000000006844: BEAD000B
	s_mul_u64 s[42:43], s[42:43], s[20:21]                     // 000000006848: AAAA142A
	s_wait_dscnt 0x0                                           // 00000000684C: BFC60000
	s_wait_alu 0xfffe                                          // 000000006850: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v19, s40                          // 000000006854: D7006A05 00005113
	s_or_b32 s46, s10, 3                                       // 00000000685C: 8C2E830A
	s_mov_b32 s47, s11                                         // 000000006860: BEAF000B
	s_add_nc_u64 s[44:45], s[34:35], s[44:45]                  // 000000006864: A9AC2C22
	s_wait_alu 0xfffd                                          // 000000006868: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s41, v20, vcc_lo             // 00000000686C: D5207C06 01AA2829
	s_lshl_b64 s[40:41], s[42:43], 2                           // 000000006874: 84A8822A
	s_add_nc_u64 s[46:47], s[34:35], s[46:47]                  // 000000006878: A9AE2E22
	s_wait_alu 0xfffe                                          // 00000000687C: BF88FFFE
	s_mul_u64 s[44:45], s[44:45], s[24:25]                     // 000000006880: AAAC182C
	v_add_co_u32 v23, vcc_lo, v19, s40                         // 000000006884: D7006A17 00005113
	s_mul_u64 s[46:47], s[46:47], s[24:25]                     // 00000000688C: AAAE182E
	s_wait_alu 0xfffe                                          // 000000006890: BF88FFFE
	s_add_nc_u64 s[44:45], s[44:45], s[22:23]                  // 000000006894: A9AC162C
	s_wait_alu 0xfffd                                          // 000000006898: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s41, v20, vcc_lo            // 00000000689C: D5207C18 01AA2829
	s_add_nc_u64 s[46:47], s[46:47], s[22:23]                  // 0000000068A4: A9AE162E
	s_wait_alu 0xfffe                                          // 0000000068A8: BF88FFFE
	s_mul_u64 s[44:45], s[44:45], s[20:21]                     // 0000000068AC: AAAC142C
	s_clause 0x1                                               // 0000000068B0: BF850001
	global_load_b32 v7, v[5:6], off                            // 0000000068B4: EE05007C 00000007 00000005
	global_load_b32 v27, v[23:24], off                         // 0000000068C0: EE05007C 0000001B 00000017
	s_mul_u64 s[46:47], s[46:47], s[20:21]                     // 0000000068CC: AAAE142E
	s_wait_alu 0xfffe                                          // 0000000068D0: BF88FFFE
	s_lshl_b64 s[42:43], s[44:45], 2                           // 0000000068D4: 84AA822C
	s_lshl_b64 s[44:45], s[46:47], 2                           // 0000000068D8: 84AC822E
	s_wait_alu 0xfffe                                          // 0000000068DC: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v19, s42                          // 0000000068E0: D7006A05 00005513
	s_wait_alu 0xfffd                                          // 0000000068E8: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s43, v20, vcc_lo             // 0000000068EC: D5207C06 01AA282B
	v_add_co_u32 v23, vcc_lo, v19, s44                         // 0000000068F4: D7006A17 00005913
	s_wait_alu 0xfffd                                          // 0000000068FC: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s45, v20, vcc_lo            // 000000006900: D5207C18 01AA282D
	s_clause 0x1                                               // 000000006908: BF850001
	global_load_b32 v5, v[5:6], off                            // 00000000690C: EE05007C 00000005 00000005
	global_load_b32 v6, v[23:24], off                          // 000000006918: EE05007C 00000006 00000017
	v_mov_b32_e32 v23, s16                                     // 000000006924: 7E2E0210
	s_add_co_i32 s10, s10, 4                                   // 000000006928: 810A840A
	s_add_co_i32 s16, s16, 16                                  // 00000000692C: 81109010
	s_wait_alu 0xfffe                                          // 000000006930: BF88FFFE
	s_cmp_eq_u32 s12, s10                                      // 000000006934: BF060A0C
	ds_load_b128 v[23:26], v23                                 // 000000006938: DBFC0000 17000017
	s_wait_loadcnt_dscnt 0x300                                 // 000000006940: BFC80300
	v_fmac_f32_e32 v4, v23, v7                                 // 000000006944: 56080F17
	s_wait_loadcnt 0x2                                         // 000000006948: BFC00002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000694C: BF8700A1
	v_fmac_f32_e32 v4, v24, v27                                // 000000006950: 56083718
	s_wait_loadcnt 0x1                                         // 000000006954: BFC00001
	v_fmac_f32_e32 v4, v25, v5                                 // 000000006958: 56080B19
	s_wait_loadcnt 0x0                                         // 00000000695C: BFC00000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006960: BF870001
	v_fmac_f32_e32 v4, v26, v6                                 // 000000006964: 56080D1A
	s_cbranch_scc0 65446                                       // 000000006968: BFA1FFA6 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xf04>
	s_and_b32 s13, s13, 3                                      // 00000000696C: 8B0D830D
	s_wait_alu 0xfffe                                          // 000000006970: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 000000006974: BF06800D
	s_cbranch_scc0 35                                          // 000000006978: BFA10023 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1108>
	s_branch 69                                                // 00000000697C: BFA00045 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1194>
	s_wait_alu 0xfffe                                          // 000000006980: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000006984: 8C7E0A7E
	s_lshr_b32 s10, s9, 1                                      // 000000006988: 850A8109
	s_cmp_gt_u32 s9, 1                                         // 00000000698C: BF088109
	s_wait_alu 0xfffe                                          // 000000006990: BF88FFFE
	s_mov_b32 s9, s10                                          // 000000006994: BE89000A
	s_wait_loadcnt_dscnt 0x0                                   // 000000006998: BFC80000
	s_barrier_signal -1                                        // 00000000699C: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000069A0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000069A4: EE0AC07C 00040000 00000000
	s_cbranch_scc0 65394                                       // 0000000069B0: BFA1FF72 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0xe7c>
	s_mov_b32 s10, exec_lo                                     // 0000000069B4: BE8A007E
	s_wait_alu 0xfffe                                          // 0000000069B8: BF88FFFE
	v_cmpx_gt_u32_e64 s9, v0                                   // 0000000069BC: D4CC007E 00020009
	s_cbranch_execz 65518                                      // 0000000069C4: BFA5FFEE <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1080>
	v_lshl_add_u32 v4, s9, 2, v17                              // 0000000069C8: D6460004 04450409
	ds_load_b32 v4, v4                                         // 0000000069D0: D8D80000 04000004
	ds_load_b32 v5, v17                                        // 0000000069D8: D8D80000 05000011
	s_wait_dscnt 0x0                                           // 0000000069E0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000069E4: 06080B04
	ds_store_b32 v17, v4                                       // 0000000069E8: D8340000 00000411
	s_branch 65507                                             // 0000000069F0: BFA0FFE3 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1080>
	s_mov_b32 s12, 0                                           // 0000000069F4: BE8C0080
	s_and_b32 s13, s13, 3                                      // 0000000069F8: 8B0D830D
	s_wait_alu 0xfffe                                          // 0000000069FC: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 000000006A00: BF06800D
	s_cbranch_scc1 35                                          // 000000006A04: BFA20023 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1194>
	s_lshl_b32 s10, s12, 2                                     // 000000006A08: 840A820C
	s_wait_alu 0xfffe                                          // 000000006A0C: BF88FFFE
	s_or_b32 s16, s10, 0x400                                   // 000000006A10: 8C10FF0A 00000400
	s_mov_b32 s10, s12                                         // 000000006A18: BE8A000C
	s_wait_alu 0xfffe                                          // 000000006A1C: BF88FFFE
	s_add_nc_u64 s[40:41], s[34:35], s[10:11]                  // 000000006A20: A9A80A22
	s_add_co_i32 s13, s13, -1                                  // 000000006A24: 810DC10D
	s_wait_alu 0xfffe                                          // 000000006A28: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[24:25]                     // 000000006A2C: AAA81828
	s_add_co_i32 s10, s10, 1                                   // 000000006A30: 810A810A
	s_wait_alu 0xfffe                                          // 000000006A34: BF88FFFE
	s_add_nc_u64 s[40:41], s[40:41], s[22:23]                  // 000000006A38: A9A81628
	s_wait_alu 0xfffe                                          // 000000006A3C: BF88FFFE
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 000000006A40: AAA81428
	s_wait_alu 0xfffe                                          // 000000006A44: BF88FFFE
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000006A48: 84A88228
	s_wait_dscnt 0x0                                           // 000000006A4C: BFC60000
	s_wait_alu 0xfffe                                          // 000000006A50: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v19, s40                          // 000000006A54: D7006A05 00005113
	s_wait_alu 0xfffd                                          // 000000006A5C: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s41, v20, vcc_lo             // 000000006A60: D5207C06 01AA2829
	global_load_b32 v5, v[5:6], off                            // 000000006A68: EE05007C 00000005 00000005
	v_mov_b32_e32 v6, s16                                      // 000000006A74: 7E0C0210
	s_add_co_i32 s16, s16, 4                                   // 000000006A78: 81108410
	s_cmp_lg_u32 s13, 0                                        // 000000006A7C: BF07800D
	ds_load_b32 v6, v6                                         // 000000006A80: D8D80000 06000006
	s_wait_loadcnt_dscnt 0x0                                   // 000000006A88: BFC80000
	v_fmac_f32_e32 v4, v6, v5                                  // 000000006A8C: 56080B06
	s_cbranch_scc1 65506                                       // 000000006A90: BFA2FFE2 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x111c>
	s_wait_dscnt 0x0                                           // 000000006A94: BFC60000
	ds_load_b32 v5, v18 offset:1280                            // 000000006A98: D8D80500 05000012
	s_wait_dscnt 0x0                                           // 000000006AA0: BFC60000
	v_fmac_f32_e32 v4, v11, v5                                 // 000000006AA4: 56080B0B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006AA8: BF870001
	v_mov_b32_e32 v11, v4                                      // 000000006AAC: 7E160304
	s_wait_alu 0xfffe                                          // 000000006AB0: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000006AB4: 8C7E097E
	ds_load_b32 v4, v18 offset:1292                            // 000000006AB8: D8D8050C 04000012
	s_add_nc_u64 s[34:35], s[34:35], 64                        // 000000006AC0: A9A2C022
	s_wait_loadcnt_dscnt 0x0                                   // 000000006AC4: BFC80000
	s_wait_alu 0xfffe                                          // 000000006AC8: BF88FFFE
	v_cmp_ge_u64_e64 s9, s[34:35], s[18:19]                    // 000000006ACC: D45E0009 00002422
	s_barrier_signal -1                                        // 000000006AD4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006AD8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000006ADC: EE0AC07C 00040000 00000000
	s_and_b32 vcc_lo, exec_lo, s9                              // 000000006AE8: 8B6A097E
	v_fmac_f32_e32 v4, v22, v5                                 // 000000006AEC: 56080B16
	s_wait_alu 0xfffe                                          // 000000006AF0: BF88FFFE
	s_cbranch_vccnz 3                                          // 000000006AF4: BFA40003 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1204>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006AF8: BF870001
	v_mov_b32_e32 v22, v4                                      // 000000006AFC: 7E2C0304
	s_branch 64860                                             // 000000006B00: BFA0FD5C <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x774>
	s_and_saveexec_b32 s3, s2                                  // 000000006B04: BE832002
	s_cbranch_execz 34                                         // 000000006B08: BFA50022 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x1294>
	v_div_scale_f32 v0, null, v4, v4, v11                      // 000000006B0C: D6FC7C00 042E0904
	s_load_b64 s[0:1], s[0:1], 0x50                            // 000000006B14: F4002000 F8000050
	s_mul_u64 s[2:3], s[20:21], s[28:29]                       // 000000006B1C: AA821C14
	s_wait_alu 0xfffe                                          // 000000006B20: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000006B24: 84828202
	v_rcp_f32_e32 v1, v0                                       // 000000006B28: 7E025500
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006B2C: BF870095
	v_fma_f32 v2, -v0, v1, 1.0                                 // 000000006B30: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 000000006B38: 56020302
	v_div_scale_f32 v2, vcc_lo, v11, v4, v11                   // 000000006B3C: D6FC6A02 042E090B
	s_wait_kmcnt 0x0                                           // 000000006B44: BFC70000
	s_wait_alu 0xfffe                                          // 000000006B48: BF88FFFE
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000006B4C: A9800200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006B50: BF870091
	v_mul_f32_e32 v3, v2, v1                                   // 000000006B54: 10060302
	v_fma_f32 v5, -v0, v3, v2                                  // 000000006B58: D6130005 240A0700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006B60: BF870091
	v_fmac_f32_e32 v3, v5, v1                                  // 000000006B64: 56060305
	v_fma_f32 v0, -v0, v3, v2                                  // 000000006B68: D6130000 240A0700
	s_wait_alu 0xfffd                                          // 000000006B70: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006B74: BF870091
	v_div_fmas_f32 v0, v0, v1, v3                              // 000000006B78: D6370000 040E0300
	v_div_fixup_f32 v0, v0, v4, v11                            // 000000006B80: D6270000 042E0900
	global_store_b32 v8, v0, s[0:1]                            // 000000006B88: EE068000 00000000 00000008
	s_endpgm                                                   // 000000006B94: BFB00000
	s_branch 64539                                             // 000000006B98: BFA0FC1B <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x308>
	s_branch 64706                                             // 000000006B9C: BFA0FCC2 <ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel+0x5a8>
	s_nop 0                                                    // 000000006BA0: BF800000
	s_nop 0                                                    // 000000006BA4: BF800000
	s_nop 0                                                    // 000000006BA8: BF800000
	s_nop 0                                                    // 000000006BAC: BF800000
	s_nop 0                                                    // 000000006BB0: BF800000
	s_nop 0                                                    // 000000006BB4: BF800000
	s_nop 0                                                    // 000000006BB8: BF800000
	s_nop 0                                                    // 000000006BBC: BF800000
	s_nop 0                                                    // 000000006BC0: BF800000
	s_nop 0                                                    // 000000006BC4: BF800000
	s_nop 0                                                    // 000000006BC8: BF800000
	s_nop 0                                                    // 000000006BCC: BF800000
	s_nop 0                                                    // 000000006BD0: BF800000
	s_nop 0                                                    // 000000006BD4: BF800000
	s_nop 0                                                    // 000000006BD8: BF800000
	s_nop 0                                                    // 000000006BDC: BF800000
	s_nop 0                                                    // 000000006BE0: BF800000
	s_nop 0                                                    // 000000006BE4: BF800000
	s_nop 0                                                    // 000000006BE8: BF800000
	s_nop 0                                                    // 000000006BEC: BF800000
	s_nop 0                                                    // 000000006BF0: BF800000
	s_nop 0                                                    // 000000006BF4: BF800000
	s_nop 0                                                    // 000000006BF8: BF800000
	s_nop 0                                                    // 000000006BFC: BF800000

0000000000006c00 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel>:
	s_load_b512 s[12:27], s[0:1], 0x0                          // 000000006C00: F4008300 F8000000
	s_mov_b32 s31, 0                                           // 000000006C08: BE9F0080
	s_mov_b32 s28, ttmp9                                       // 000000006C0C: BE9C0075
	s_mov_b32 s29, s31                                         // 000000006C10: BE9D001F
	s_wait_kmcnt 0x0                                           // 000000006C14: BFC70000
	s_mul_u64 s[2:3], s[22:23], s[20:21]                       // 000000006C18: AA821416
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006C1C: BF870009
	v_cmp_le_u64_e64 s2, s[2:3], s[28:29]                      // 000000006C20: D45B0002 00003802
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000006C28: 8B6A027E
	s_cbranch_vccnz 1207                                       // 000000006C2C: BFA404B7 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x130c>
	s_clause 0x1                                               // 000000006C30: BF850001
	s_load_b32 s2, s[0:1], 0x64                                // 000000006C34: F4000080 F8000064
	s_load_b64 s[20:21], s[0:1], 0x40                          // 000000006C3C: F4002500 F8000040
	s_wait_kmcnt 0x0                                           // 000000006C44: BFC70000
	s_and_b32 s30, s2, 0xffff                                  // 000000006C48: 8B1EFF02 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006C50: BF870009
	v_cmp_gt_u64_e64 s2, s[20:21], s[30:31]                    // 000000006C54: D45C0002 00003C14
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000006C5C: 8B6A027E
	s_cbranch_vccnz 1194                                       // 000000006C60: BFA404AA <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x130c>
	v_cmp_lt_u64_e64 s2, s[28:29], s[22:23]                    // 000000006C64: D4590002 00002C1C
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000006C6C: 8B6A027E
	s_mov_b64 s[2:3], 0                                        // 000000006C70: BE820180
	s_cbranch_vccnz 32                                         // 000000006C74: BFA40020 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xf8>
	v_cvt_f32_u32_e32 v1, s22                                  // 000000006C78: 7E020C16
	s_sub_co_i32 s3, 0, s22                                    // 000000006C7C: 81831680
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000006C80: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000006C84: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000006C88: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006C90: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000006C94: 7E020F01
	v_readfirstlane_b32 s2, v1                                 // 000000006C98: 7E040501
	s_wait_alu 0xfffe                                          // 000000006C9C: BF88FFFE
	s_mul_i32 s3, s3, s2                                       // 000000006CA0: 96030203
	s_wait_alu 0xfffe                                          // 000000006CA4: BF88FFFE
	s_mul_hi_u32 s3, s2, s3                                    // 000000006CA8: 96830302
	s_wait_alu 0xfffe                                          // 000000006CAC: BF88FFFE
	s_add_co_i32 s2, s2, s3                                    // 000000006CB0: 81020302
	s_wait_alu 0xfffe                                          // 000000006CB4: BF88FFFE
	s_mul_hi_u32 s2, s28, s2                                   // 000000006CB8: 9682021C
	s_wait_alu 0xfffe                                          // 000000006CBC: BF88FFFE
	s_mul_i32 s3, s2, s22                                      // 000000006CC0: 96031602
	s_add_co_i32 s4, s2, 1                                     // 000000006CC4: 81048102
	s_wait_alu 0xfffe                                          // 000000006CC8: BF88FFFE
	s_sub_co_i32 s3, s28, s3                                   // 000000006CCC: 8183031C
	s_wait_alu 0xfffe                                          // 000000006CD0: BF88FFFE
	s_sub_co_i32 s5, s3, s22                                   // 000000006CD4: 81851603
	s_cmp_ge_u32 s3, s22                                       // 000000006CD8: BF091603
	s_cselect_b32 s2, s4, s2                                   // 000000006CDC: 98020204
	s_cselect_b32 s3, s5, s3                                   // 000000006CE0: 98030305
	s_wait_alu 0xfffe                                          // 000000006CE4: BF88FFFE
	s_add_co_i32 s4, s2, 1                                     // 000000006CE8: 81048102
	s_cmp_ge_u32 s3, s22                                       // 000000006CEC: BF091603
	s_mov_b32 s3, 0                                            // 000000006CF0: BE830080
	s_cselect_b32 s2, s4, s2                                   // 000000006CF4: 98020204
	s_or_b64 s[6:7], s[22:23], s[24:25]                        // 000000006CF8: 8C861816
	s_mov_b32 s6, 0                                            // 000000006CFC: BE860080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006D00: BF870009
	s_cmp_lg_u64 s[6:7], 0                                     // 000000006D04: BF118006
	s_cbranch_scc0 1153                                        // 000000006D08: BFA10481 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x1310>
	s_cvt_f32_u32 s4, s24                                      // 000000006D0C: BE846518
	s_cvt_f32_u32 s5, s25                                      // 000000006D10: BE856519
	s_sub_nc_u64 s[8:9], 0, s[24:25]                           // 000000006D14: AA081880
	s_mov_b32 s11, s6                                          // 000000006D18: BE8B0006
	s_mov_b32 s37, s6                                          // 000000006D1C: BEA50006
	s_fmamk_f32 s4, s5, 0x4f800000, s4                         // 000000006D20: A3040405 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000006D28: BF87029B
	v_s_rcp_f32 s4, s4                                         // 000000006D2C: D6840004 00000004
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000006D34: A204FF04 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000006D3C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000006D40: BF87052A
	s_mul_f32 s5, s4, 0x2f800000                               // 000000006D44: A205FF04 2F800000
	s_wait_alu 0xfffe                                          // 000000006D4C: BF88FFFE
	s_trunc_f32 s5, s5                                         // 000000006D50: BE856205
	s_wait_alu 0xfffe                                          // 000000006D54: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000006D58: BF8704BA
	s_fmamk_f32 s4, s5, 0xcf800000, s4                         // 000000006D5C: A3040405 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000006D64: BE856705
	s_wait_alu 0xfffe                                          // 000000006D68: BF88FFFE
	s_cvt_u32_f32 s4, s4                                       // 000000006D6C: BE846704
	s_wait_alu 0xfffe                                          // 000000006D70: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000006D74: BF87049A
	s_mul_u64 s[34:35], s[8:9], s[4:5]                         // 000000006D78: AAA20408
	s_mul_hi_u32 s39, s4, s35                                  // 000000006D7C: 96A72304
	s_mul_i32 s38, s4, s35                                     // 000000006D80: 96262304
	s_mul_hi_u32 s10, s4, s34                                  // 000000006D84: 968A2204
	s_mul_i32 s31, s5, s34                                     // 000000006D88: 961F2205
	s_add_nc_u64 s[10:11], s[10:11], s[38:39]                  // 000000006D8C: A98A260A
	s_mul_hi_u32 s7, s5, s34                                   // 000000006D90: 96872205
	s_mul_hi_u32 s33, s5, s35                                  // 000000006D94: 96A12305
	s_wait_alu 0xfffe                                          // 000000006D98: BF88FFFE
	s_add_co_u32 s10, s10, s31                                 // 000000006D9C: 800A1F0A
	s_add_co_ci_u32 s36, s11, s7                               // 000000006DA0: 8224070B
	s_mul_i32 s34, s5, s35                                     // 000000006DA4: 96222305
	s_add_co_ci_u32 s35, s33, 0                                // 000000006DA8: 82238021
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000006DAC: BF8704C9
	s_add_nc_u64 s[10:11], s[36:37], s[34:35]                  // 000000006DB0: A98A2224
	s_mov_b32 s35, s6                                          // 000000006DB4: BEA30006
	s_add_co_u32 s4, s4, s10                                   // 000000006DB8: 80040A04
	s_cselect_b32 s7, -1, 0                                    // 000000006DBC: 980780C1
	s_cmp_lg_u32 s7, 0                                         // 000000006DC0: BF078007
	s_add_co_ci_u32 s5, s5, s11                                // 000000006DC4: 82050B05
	s_mov_b32 s11, s6                                          // 000000006DC8: BE8B0006
	s_wait_alu 0xfffe                                          // 000000006DCC: BF88FFFE
	s_mul_u64 s[8:9], s[8:9], s[4:5]                           // 000000006DD0: AA880408
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006DD4: BF870009
	s_mul_hi_u32 s37, s4, s9                                   // 000000006DD8: 96A50904
	s_mul_i32 s36, s4, s9                                      // 000000006DDC: 96240904
	s_mul_hi_u32 s10, s4, s8                                   // 000000006DE0: 968A0804
	s_mul_i32 s31, s5, s8                                      // 000000006DE4: 961F0805
	s_add_nc_u64 s[10:11], s[10:11], s[36:37]                  // 000000006DE8: A98A240A
	s_mul_hi_u32 s7, s5, s8                                    // 000000006DEC: 96870805
	s_mul_hi_u32 s33, s5, s9                                   // 000000006DF0: 96A10905
	s_mul_i32 s8, s5, s9                                       // 000000006DF4: 96080905
	s_wait_alu 0xfffe                                          // 000000006DF8: BF88FFFE
	s_add_co_u32 s9, s10, s31                                  // 000000006DFC: 80091F0A
	s_add_co_ci_u32 s34, s11, s7                               // 000000006E00: 8222070B
	s_add_co_ci_u32 s9, s33, 0                                 // 000000006E04: 82098021
	s_mov_b32 s11, s6                                          // 000000006E08: BE8B0006
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]                      // 000000006E0C: A9880822
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006E10: BF870009
	s_add_co_u32 s4, s4, s8                                    // 000000006E14: 80040804
	s_cselect_b32 s7, -1, 0                                    // 000000006E18: 980780C1
	s_wait_alu 0xfffe                                          // 000000006E1C: BF88FFFE
	s_mul_hi_u32 s10, s22, s4                                  // 000000006E20: 968A0416
	s_cmp_lg_u32 s7, 0                                         // 000000006E24: BF078007
	s_mul_hi_u32 s7, s23, s4                                   // 000000006E28: 96870417
	s_add_co_ci_u32 s8, s5, s9                                 // 000000006E2C: 82080905
	s_mul_i32 s9, s23, s4                                      // 000000006E30: 96090417
	s_mul_hi_u32 s5, s22, s8                                   // 000000006E34: 96850816
	s_mul_i32 s4, s22, s8                                      // 000000006E38: 96040816
	s_mul_hi_u32 s31, s23, s8                                  // 000000006E3C: 969F0817
	s_wait_alu 0xfffe                                          // 000000006E40: BF88FFFE
	s_add_nc_u64 s[4:5], s[10:11], s[4:5]                      // 000000006E44: A984040A
	s_mul_i32 s8, s23, s8                                      // 000000006E48: 96080817
	s_wait_alu 0xfffe                                          // 000000006E4C: BF88FFFE
	s_add_co_u32 s4, s4, s9                                    // 000000006E50: 80040904
	s_add_co_ci_u32 s34, s5, s7                                // 000000006E54: 82220705
	s_add_co_ci_u32 s9, s31, 0                                 // 000000006E58: 8209801F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006E5C: BF870009
	s_add_nc_u64 s[4:5], s[34:35], s[8:9]                      // 000000006E60: A9840822
	s_wait_alu 0xfffe                                          // 000000006E64: BF88FFFE
	s_mul_u64 s[8:9], s[24:25], s[4:5]                         // 000000006E68: AA880418
	s_add_nc_u64 s[34:35], s[4:5], 2                           // 000000006E6C: A9A28204
	s_sub_co_u32 s7, s22, s8                                   // 000000006E70: 80870816
	s_cselect_b32 s8, -1, 0                                    // 000000006E74: 980880C1
	s_sub_co_i32 s10, s23, s9                                  // 000000006E78: 818A0917
	s_cmp_lg_u32 s8, 0                                         // 000000006E7C: BF078008
	s_sub_co_ci_u32 s10, s10, s25                              // 000000006E80: 828A190A
	s_sub_co_u32 s11, s7, s24                                  // 000000006E84: 808B1807
	s_cselect_b32 s31, -1, 0                                   // 000000006E88: 981F80C1
	s_wait_alu 0xfffe                                          // 000000006E8C: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000006E90: BF07801F
	s_sub_co_ci_u32 s10, s10, 0                                // 000000006E94: 828A800A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006E98: BF870009
	s_cmp_ge_u32 s10, s25                                      // 000000006E9C: BF09190A
	s_cselect_b32 s31, -1, 0                                   // 000000006EA0: 981F80C1
	s_cmp_ge_u32 s11, s24                                      // 000000006EA4: BF09180B
	s_cselect_b32 s33, -1, 0                                   // 000000006EA8: 982180C1
	s_cmp_eq_u32 s10, s25                                      // 000000006EAC: BF06190A
	s_add_nc_u64 s[10:11], s[4:5], 1                           // 000000006EB0: A98A8104
	s_wait_alu 0xfffe                                          // 000000006EB4: BF88FFFE
	s_cselect_b32 s31, s33, s31                                // 000000006EB8: 981F1F21
	s_wait_alu 0xfffe                                          // 000000006EBC: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 000000006EC0: BF07801F
	s_cselect_b32 s10, s34, s10                                // 000000006EC4: 980A0A22
	s_cselect_b32 s11, s35, s11                                // 000000006EC8: 980B0B23
	s_cmp_lg_u32 s8, 0                                         // 000000006ECC: BF078008
	s_sub_co_ci_u32 s8, s23, s9                                // 000000006ED0: 82880917
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006ED4: BF870009
	s_cmp_ge_u32 s8, s25                                       // 000000006ED8: BF091908
	s_cselect_b32 s9, -1, 0                                    // 000000006EDC: 980980C1
	s_cmp_ge_u32 s7, s24                                       // 000000006EE0: BF091807
	s_cselect_b32 s7, -1, 0                                    // 000000006EE4: 980780C1
	s_cmp_eq_u32 s8, s25                                       // 000000006EE8: BF061908
	s_cselect_b32 s7, s7, s9                                   // 000000006EEC: 98070907
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006EF0: BF870009
	s_cmp_lg_u32 s7, 0                                         // 000000006EF4: BF078007
	s_cselect_b32 s5, s11, s5                                  // 000000006EF8: 9805050B
	s_cselect_b32 s4, s10, s4                                  // 000000006EFC: 9804040A
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000006F00: 916A067E
	s_cbranch_vccnz 32                                         // 000000006F04: BFA40020 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x388>
	v_cvt_f32_u32_e32 v1, s24                                  // 000000006F08: 7E020C18
	s_sub_co_i32 s5, 0, s24                                    // 000000006F0C: 81851880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000006F10: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000006F14: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000006F18: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F20: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000006F24: 7E020F01
	v_readfirstlane_b32 s4, v1                                 // 000000006F28: 7E080501
	s_wait_alu 0xfffe                                          // 000000006F2C: BF88FFFE
	s_mul_i32 s5, s5, s4                                       // 000000006F30: 96050405
	s_wait_alu 0xfffe                                          // 000000006F34: BF88FFFE
	s_mul_hi_u32 s5, s4, s5                                    // 000000006F38: 96850504
	s_wait_alu 0xfffe                                          // 000000006F3C: BF88FFFE
	s_add_co_i32 s4, s4, s5                                    // 000000006F40: 81040504
	s_wait_alu 0xfffe                                          // 000000006F44: BF88FFFE
	s_mul_hi_u32 s4, s22, s4                                   // 000000006F48: 96840416
	s_wait_alu 0xfffe                                          // 000000006F4C: BF88FFFE
	s_mul_i32 s5, s4, s24                                      // 000000006F50: 96051804
	s_add_co_i32 s6, s4, 1                                     // 000000006F54: 81068104
	s_wait_alu 0xfffe                                          // 000000006F58: BF88FFFE
	s_sub_co_i32 s5, s22, s5                                   // 000000006F5C: 81850516
	s_wait_alu 0xfffe                                          // 000000006F60: BF88FFFE
	s_sub_co_i32 s7, s5, s24                                   // 000000006F64: 81871805
	s_cmp_ge_u32 s5, s24                                       // 000000006F68: BF091805
	s_cselect_b32 s4, s6, s4                                   // 000000006F6C: 98040406
	s_cselect_b32 s5, s7, s5                                   // 000000006F70: 98050507
	s_wait_alu 0xfffe                                          // 000000006F74: BF88FFFE
	s_add_co_i32 s6, s4, 1                                     // 000000006F78: 81068104
	s_cmp_ge_u32 s5, s24                                       // 000000006F7C: BF091805
	s_mov_b32 s5, 0                                            // 000000006F80: BE850080
	s_cselect_b32 s4, s6, s4                                   // 000000006F84: 98040406
	s_mul_u64 s[6:7], s[2:3], s[22:23]                         // 000000006F88: AA861602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000006F8C: BF8704C9
	s_sub_nc_u64 s[6:7], s[28:29], s[6:7]                      // 000000006F90: AA06061C
	s_wait_alu 0xfffe                                          // 000000006F94: BF88FFFE
	s_or_b64 s[8:9], s[6:7], s[4:5]                            // 000000006F98: 8C880406
	s_mov_b32 s8, 0                                            // 000000006F9C: BE880080
	s_cmp_lg_u64 s[8:9], 0                                     // 000000006FA0: BF118008
	s_cbranch_scc0 987                                         // 000000006FA4: BFA103DB <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x1314>
	s_cvt_f32_u32 s9, s4                                       // 000000006FA8: BE896504
	s_cvt_f32_u32 s10, s5                                      // 000000006FAC: BE8A6505
	s_sub_nc_u64 s[22:23], 0, s[4:5]                           // 000000006FB0: AA160480
	s_mov_b32 s35, s8                                          // 000000006FB4: BEA30008
	s_mov_b32 s39, s8                                          // 000000006FB8: BEA70008
	s_fmamk_f32 s9, s10, 0x4f800000, s9                        // 000000006FBC: A309090A 4F800000
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000006FC4: BF87029B
	v_s_rcp_f32 s9, s9                                         // 000000006FC8: D6840009 00000009
	s_mul_f32 s9, s9, 0x5f7ffffc                               // 000000006FD0: A209FF09 5F7FFFFC
	s_wait_alu 0xfffe                                          // 000000006FD8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000006FDC: BF87059A
	s_mul_f32 s10, s9, 0x2f800000                              // 000000006FE0: A20AFF09 2F800000
	s_trunc_f32 s10, s10                                       // 000000006FE8: BE8A620A
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000006FEC: BF8704BB
	s_fmamk_f32 s9, s10, 0xcf800000, s9                        // 000000006FF0: A309090A CF800000
	s_cvt_u32_f32 s11, s10                                     // 000000006FF8: BE8B670A
	s_wait_alu 0xfffe                                          // 000000006FFC: BF88FFFE
	s_cvt_u32_f32 s10, s9                                      // 000000007000: BE8A6709
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000007004: BF87049B
	s_mul_u64 s[36:37], s[22:23], s[10:11]                     // 000000007008: AAA40A16
	s_mul_hi_u32 s41, s10, s37                                 // 00000000700C: 96A9250A
	s_mul_i32 s40, s10, s37                                    // 000000007010: 9628250A
	s_mul_hi_u32 s34, s10, s36                                 // 000000007014: 96A2240A
	s_mul_i32 s31, s11, s36                                    // 000000007018: 961F240B
	s_add_nc_u64 s[34:35], s[34:35], s[40:41]                  // 00000000701C: A9A22822
	s_mul_hi_u32 s9, s11, s36                                  // 000000007020: 9689240B
	s_mul_hi_u32 s33, s11, s37                                 // 000000007024: 96A1250B
	s_wait_alu 0xfffe                                          // 000000007028: BF88FFFE
	s_add_co_u32 s31, s34, s31                                 // 00000000702C: 801F1F22
	s_add_co_ci_u32 s38, s35, s9                               // 000000007030: 82260923
	s_mul_i32 s36, s11, s37                                    // 000000007034: 9624250B
	s_add_co_ci_u32 s37, s33, 0                                // 000000007038: 82258021
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000703C: BF870009
	s_add_nc_u64 s[34:35], s[38:39], s[36:37]                  // 000000007040: A9A22426
	s_mov_b32 s37, s8                                          // 000000007044: BEA50008
	s_add_co_u32 s10, s10, s34                                 // 000000007048: 800A220A
	s_cselect_b32 s9, -1, 0                                    // 00000000704C: 980980C1
	s_wait_alu 0xfffe                                          // 000000007050: BF88FFFE
	s_cmp_lg_u32 s9, 0                                         // 000000007054: BF078009
	s_add_co_ci_u32 s11, s11, s35                              // 000000007058: 820B230B
	s_mov_b32 s35, s8                                          // 00000000705C: BEA30008
	s_mul_u64 s[22:23], s[22:23], s[10:11]                     // 000000007060: AA960A16
	s_wait_alu 0xfffe                                          // 000000007064: BF88FFFE
	s_mul_hi_u32 s39, s10, s23                                 // 000000007068: 96A7170A
	s_mul_i32 s38, s10, s23                                    // 00000000706C: 9626170A
	s_mul_hi_u32 s34, s10, s22                                 // 000000007070: 96A2160A
	s_mul_i32 s31, s11, s22                                    // 000000007074: 961F160B
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]                  // 000000007078: A9A22622
	s_mul_hi_u32 s9, s11, s22                                  // 00000000707C: 9689160B
	s_mul_hi_u32 s33, s11, s23                                 // 000000007080: 96A1170B
	s_mul_i32 s22, s11, s23                                    // 000000007084: 9616170B
	s_wait_alu 0xfffe                                          // 000000007088: BF88FFFE
	s_add_co_u32 s23, s34, s31                                 // 00000000708C: 80171F22
	s_add_co_ci_u32 s36, s35, s9                               // 000000007090: 82240923
	s_add_co_ci_u32 s23, s33, 0                                // 000000007094: 82178021
	s_mov_b32 s35, s8                                          // 000000007098: BEA30008
	s_wait_alu 0xfffe                                          // 00000000709C: BF88FFFE
	s_add_nc_u64 s[22:23], s[36:37], s[22:23]                  // 0000000070A0: A9961624
	s_wait_alu 0xfffe                                          // 0000000070A4: BF88FFFE
	s_add_co_u32 s9, s10, s22                                  // 0000000070A8: 8009160A
	s_cselect_b32 s10, -1, 0                                   // 0000000070AC: 980A80C1
	s_wait_alu 0xfffe                                          // 0000000070B0: BF88FFFE
	s_mul_hi_u32 s34, s6, s9                                   // 0000000070B4: 96A20906
	s_cmp_lg_u32 s10, 0                                        // 0000000070B8: BF07800A
	s_mul_hi_u32 s31, s7, s9                                   // 0000000070BC: 969F0907
	s_add_co_ci_u32 s22, s11, s23                              // 0000000070C0: 8216170B
	s_mul_i32 s9, s7, s9                                       // 0000000070C4: 96090907
	s_wait_alu 0xfffe                                          // 0000000070C8: BF88FFFE
	s_mul_hi_u32 s11, s6, s22                                  // 0000000070CC: 968B1606
	s_mul_i32 s10, s6, s22                                     // 0000000070D0: 960A1606
	s_mul_hi_u32 s23, s7, s22                                  // 0000000070D4: 96971607
	s_add_nc_u64 s[10:11], s[34:35], s[10:11]                  // 0000000070D8: A98A0A22
	s_mul_i32 s22, s7, s22                                     // 0000000070DC: 96161607
	s_add_co_u32 s9, s10, s9                                   // 0000000070E0: 8009090A
	s_add_co_ci_u32 s36, s11, s31                              // 0000000070E4: 82241F0B
	s_wait_alu 0xfffe                                          // 0000000070E8: BF88FFFE
	s_add_co_ci_u32 s23, s23, 0                                // 0000000070EC: 82178017
	s_wait_alu 0xfffe                                          // 0000000070F0: BF88FFFE
	s_add_nc_u64 s[10:11], s[36:37], s[22:23]                  // 0000000070F4: A98A1624
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000070F8: BF870009
	s_mul_u64 s[22:23], s[4:5], s[10:11]                       // 0000000070FC: AA960A04
	s_wait_alu 0xfffe                                          // 000000007100: BF88FFFE
	s_sub_co_u32 s9, s6, s22                                   // 000000007104: 80891606
	s_cselect_b32 s22, -1, 0                                   // 000000007108: 981680C1
	s_sub_co_i32 s31, s7, s23                                  // 00000000710C: 819F1707
	s_wait_alu 0xfffe                                          // 000000007110: BF88FFFE
	s_cmp_lg_u32 s22, 0                                        // 000000007114: BF078016
	s_sub_co_ci_u32 s31, s31, s5                               // 000000007118: 829F051F
	s_sub_co_u32 s33, s9, s4                                   // 00000000711C: 80A10409
	s_cselect_b32 s34, -1, 0                                   // 000000007120: 982280C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007124: BF870009
	s_cmp_lg_u32 s34, 0                                        // 000000007128: BF078022
	s_add_nc_u64 s[34:35], s[10:11], 1                         // 00000000712C: A9A2810A
	s_wait_alu 0xfffe                                          // 000000007130: BF88FFFE
	s_sub_co_ci_u32 s31, s31, 0                                // 000000007134: 829F801F
	s_wait_alu 0xfffe                                          // 000000007138: BF88FFFE
	s_cmp_ge_u32 s31, s5                                       // 00000000713C: BF09051F
	s_cselect_b32 s36, -1, 0                                   // 000000007140: 982480C1
	s_cmp_ge_u32 s33, s4                                       // 000000007144: BF090421
	s_cselect_b32 s33, -1, 0                                   // 000000007148: 982180C1
	s_cmp_eq_u32 s31, s5                                       // 00000000714C: BF06051F
	s_cselect_b32 s31, s33, s36                                // 000000007150: 981F2421
	s_add_nc_u64 s[36:37], s[10:11], 2                         // 000000007154: A9A4820A
	s_wait_alu 0xfffe                                          // 000000007158: BF88FFFE
	s_cmp_lg_u32 s31, 0                                        // 00000000715C: BF07801F
	s_cselect_b32 s31, s36, s34                                // 000000007160: 981F2224
	s_cselect_b32 s33, s37, s35                                // 000000007164: 98212325
	s_cmp_lg_u32 s22, 0                                        // 000000007168: BF078016
	s_sub_co_ci_u32 s7, s7, s23                                // 00000000716C: 82871707
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007170: BF870009
	s_cmp_ge_u32 s7, s5                                        // 000000007174: BF090507
	s_cselect_b32 s22, -1, 0                                   // 000000007178: 981680C1
	s_cmp_ge_u32 s9, s4                                        // 00000000717C: BF090409
	s_cselect_b32 s9, -1, 0                                    // 000000007180: 980980C1
	s_cmp_eq_u32 s7, s5                                        // 000000007184: BF060507
	s_wait_alu 0xfffe                                          // 000000007188: BF88FFFE
	s_cselect_b32 s5, s9, s22                                  // 00000000718C: 98051609
	s_wait_alu 0xfffe                                          // 000000007190: BF88FFFE
	s_cmp_lg_u32 s5, 0                                         // 000000007194: BF078005
	s_cselect_b32 s23, s33, s11                                // 000000007198: 98170B21
	s_cselect_b32 s22, s31, s10                                // 00000000719C: 98160A1F
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 0000000071A0: 916A087E
	s_cbranch_vccnz 29                                         // 0000000071A4: BFA4001D <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x61c>
	v_cvt_f32_u32_e32 v1, s4                                   // 0000000071A8: 7E020C04
	s_sub_co_i32 s7, 0, s4                                     // 0000000071AC: 81870480
	s_mov_b32 s23, 0                                           // 0000000071B0: BE970080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000071B4: BF870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 0000000071B8: 7E025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000071BC: 100202FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000071C4: BF870091
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000071C8: 7E020F01
	v_readfirstlane_b32 s5, v1                                 // 0000000071CC: 7E0A0501
	s_mul_i32 s7, s7, s5                                       // 0000000071D0: 96070507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000071D4: BF870499
	s_mul_hi_u32 s7, s5, s7                                    // 0000000071D8: 96870705
	s_add_co_i32 s5, s5, s7                                    // 0000000071DC: 81050705
	s_wait_alu 0xfffe                                          // 0000000071E0: BF88FFFE
	s_mul_hi_u32 s5, s6, s5                                    // 0000000071E4: 96850506
	s_wait_alu 0xfffe                                          // 0000000071E8: BF88FFFE
	s_mul_i32 s7, s5, s4                                       // 0000000071EC: 96070405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000071F0: BF870009
	s_sub_co_i32 s6, s6, s7                                    // 0000000071F4: 81860706
	s_add_co_i32 s7, s5, 1                                     // 0000000071F8: 81078105
	s_sub_co_i32 s8, s6, s4                                    // 0000000071FC: 81880406
	s_cmp_ge_u32 s6, s4                                        // 000000007200: BF090406
	s_cselect_b32 s5, s7, s5                                   // 000000007204: 98050507
	s_wait_alu 0xfffe                                          // 000000007208: BF88FFFE
	s_cselect_b32 s6, s8, s6                                   // 00000000720C: 98060608
	s_add_co_i32 s7, s5, 1                                     // 000000007210: 81078105
	s_cmp_ge_u32 s6, s4                                        // 000000007214: BF090406
	s_cselect_b32 s22, s7, s5                                  // 000000007218: 98160507
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v8, 2, v0       // 00000000721C: CA220080 01080082
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]                      // 000000007224: A9820212
	s_mov_b64 s[34:35], 0                                      // 000000007228: BEA20180
	s_wait_alu 0xfffe                                          // 00000000722C: BF88FFFE
	s_add_nc_u64 s[18:19], s[2:3], 1                           // 000000007230: A9928102
	v_cmp_gt_u64_e64 s2, s[20:21], v[0:1]                      // 000000007234: D45C0002 00020014
	v_cmp_le_u64_e64 s3, s[20:21], v[0:1]                      // 00000000723C: D45B0003 00020014
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v11, v1            // 000000007244: CA100101 040A0101
	s_cmp_eq_u64 s[18:19], 0                                   // 00000000724C: BF108012
	s_cbranch_scc1 778                                         // 000000007250: BFA2030A <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x127c>
	v_mbcnt_lo_u32_b32 v3, -1, 0                               // 000000007254: D71F0003 000100C1
	v_dual_mov_b32 v17, 0 :: v_dual_and_b32 v2, 31, v0         // 00000000725C: CA240080 1102009F
	s_load_b32 s31, s[0:1], 0x48                               // 000000007264: F40007C0 F8000048
	s_mul_u64 s[36:37], s[26:27], s[28:29]                     // 00000000726C: AAA41C1A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007270: BF870112
	v_xor_b32_e32 v4, 31, v3                                   // 000000007274: 3A08069F
	v_cmp_eq_u32_e64 s5, 0, v2                                 // 000000007278: D44A0005 00020480
	v_cmp_gt_u32_e64 s7, 8, v2                                 // 000000007280: D44C0007 00020488
	v_dual_mov_b32 v21, 0 :: v_dual_lshlrev_b32 v10, 2, v2     // 000000007288: CA220080 150A0482
	s_delay_alu instid0(VALU_DEP_4)                            // 000000007290: BF870004
	v_cmp_gt_u32_e32 vcc_lo, 8, v4                             // 000000007294: 7C980888
	v_dual_mov_b32 v11, 0 :: v_dual_and_b32 v2, 16, v4         // 000000007298: CA240080 0B020890
	s_lshl_b64 s[36:37], s[36:37], 2                           // 0000000072A0: 84A48224
	v_add_co_u32 v18, s9, s16, v8                              // 0000000072A4: D7000912 00021010
	v_cndmask_b32_e64 v5, 8, 0, vcc_lo                         // 0000000072AC: D5010005 01A90088
	v_cmp_gt_u32_e32 vcc_lo, 4, v4                             // 0000000072B4: 7C980884
	s_add_nc_u64 s[12:13], s[12:13], s[36:37]                  // 0000000072B8: A98C240C
	v_add_lshl_u32 v12, v2, v3, 2                              // 0000000072BC: D647000C 020A0702
	s_wait_alu 0xf1ff                                          // 0000000072C4: BF88F1FF
	v_add_co_ci_u32_e64 v19, null, s17, 0, s9                  // 0000000072C8: D5207C13 00250011
	s_wait_alu 0xfffd                                          // 0000000072D0: BF88FFFD
	v_cndmask_b32_e64 v6, 4, 0, vcc_lo                         // 0000000072D4: D5010006 01A90084
	v_cmp_gt_u32_e32 vcc_lo, 2, v4                             // 0000000072DC: 7C980882
	v_add_co_u32 v2, s9, s12, v8                               // 0000000072E0: D7000902 0002100C
	v_cmp_gt_u64_e64 s4, s[26:27], v[0:1]                      // 0000000072E8: D45C0004 0002001A
	v_lshrrev_b32_e32 v9, 3, v0                                // 0000000072F0: 32120083
	s_wait_alu 0xfffd                                          // 0000000072F4: BF88FFFD
	v_cndmask_b32_e64 v4, 2, 0, vcc_lo                         // 0000000072F8: D5010004 01A90082
	v_cmp_ne_u32_e32 vcc_lo, 31, v3                            // 000000007300: 7C9A069F
	v_cmp_gt_u32_e64 s6, 32, v0                                // 000000007304: D44C0006 000200A0
	v_cmp_eq_u32_e64 s8, 0, v0                                 // 00000000730C: D44A0008 00020080
	v_add_lshl_u32 v13, v5, v3, 2                              // 000000007314: D647000D 020A0705
	v_add_lshl_u32 v14, v6, v3, 2                              // 00000000731C: D647000E 020A0706
	s_wait_alu 0xfffd                                          // 000000007324: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo                // 000000007328: D5207C07 01AA0680
	v_add_lshl_u32 v15, v4, v3, 2                              // 000000007330: D647000F 020A0704
	s_wait_alu 0xf1ff                                          // 000000007338: BF88F1FF
	v_add_co_ci_u32_e64 v3, null, s13, 0, s9                   // 00000000733C: D5207C03 0025000D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007344: BF870003
	v_lshlrev_b32_e32 v16, 2, v7                               // 000000007348: 30200E82
	v_lshl_add_u32 v20, v0, 2, 0x400                           // 00000000734C: D6460014 03FD0500 00000400
	s_mov_b32 s11, 0                                           // 000000007358: BE8B0080
	s_lshl_b32 s33, s30, 2                                     // 00000000735C: 8421821E
	s_lshl_b32 s36, s30, 2                                     // 000000007360: 8424821E
	s_mov_b32 s37, 0xff7fffff                                  // 000000007364: BEA500FF FF7FFFFF
	s_sub_nc_u64 s[12:13], s[18:19], s[34:35]                  // 00000000736C: AA0C2212
	s_mov_b32 s10, s11                                         // 000000007370: BE8A000B
	s_wait_alu 0xfffe                                          // 000000007374: BF88FFFE
	v_cmp_lt_u64_e64 s9, s[12:13], 64                          // 000000007378: D4590009 0001800C
	s_and_b32 s9, s9, exec_lo                                  // 000000007380: 8B097E09
	s_cselect_b32 s38, s12, 64                                 // 000000007384: 9826C00C
	s_branch 12                                                // 000000007388: BFA0000C <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x7bc>
	s_wait_alu 0xfffe                                          // 00000000738C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007390: 8C7E097E
	s_add_co_i32 s10, s10, 1                                   // 000000007394: 810A810A
	s_wait_loadcnt_dscnt 0x0                                   // 000000007398: BFC80000
	s_wait_alu 0xfffe                                          // 00000000739C: BF88FFFE
	s_cmp_ge_u32 s10, s38                                      // 0000000073A0: BF09260A
	s_barrier_signal -1                                        // 0000000073A4: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000073A8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000073AC: EE0AC07C 00040000 00000000
	s_cbranch_scc1 141                                         // 0000000073B8: BFA2008D <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x9f0>
	v_mov_b32_e32 v22, 0                                       // 0000000073BC: 7E2C0280
	s_and_saveexec_b32 s39, s4                                 // 0000000073C0: BEA72004
	s_cbranch_execz 50                                         // 0000000073C4: BFA50032 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x890>
	s_add_nc_u64 s[16:17], s[34:35], s[10:11]                  // 0000000073C8: A9900A22
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v5, v3             // 0000000073CC: CA100080 16040103
	s_wait_alu 0xfffe                                          // 0000000073D4: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[24:25]                     // 0000000073D8: AA901810
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v7, v1             // 0000000073DC: CA100102 04060101
	s_wait_alu 0xfffe                                          // 0000000073E4: BF88FFFE
	s_add_nc_u64 s[16:17], s[16:17], s[22:23]                  // 0000000073E8: A9901610
	v_mov_b32_e32 v6, v0                                       // 0000000073EC: 7E0C0300
	s_wait_alu 0xfffe                                          // 0000000073F0: BF88FFFE
	s_mul_u64 s[16:17], s[16:17], s[26:27]                     // 0000000073F4: AA901A10
	s_mov_b32 s40, 0                                           // 0000000073F8: BEA80080
	s_wait_alu 0xfffe                                          // 0000000073FC: BF88FFFE
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000007400: 84908210
	s_wait_alu 0xfffe                                          // 000000007404: BF88FFFE
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]                  // 000000007408: A990100E
	v_lshlrev_b64_e32 v[23:24], 2, v[6:7]                      // 00000000740C: 3E2E0C82
	s_wait_alu 0xfffe                                          // 000000007410: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007414: BF870121
	v_add_co_u32 v23, vcc_lo, s16, v23                         // 000000007418: D7006A17 00022E10
	s_wait_alu 0xfffd                                          // 000000007420: BF88FFFD
	v_add_co_ci_u32_e64 v24, null, s17, v24, vcc_lo            // 000000007424: D5207C18 01AA3011
	v_add_co_u32 v6, vcc_lo, v6, s30                           // 00000000742C: D7006A06 00003D06
	global_load_b32 v25, v[4:5], off                           // 000000007434: EE05007C 00000019 00000004
	global_load_b32 v23, v[23:24], off                         // 000000007440: EE05007C 00000017 00000017
	s_wait_alu 0xfffd                                          // 00000000744C: BF88FFFD
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo                // 000000007450: D5207C07 01AA0E80
	v_add_co_u32 v4, s9, v4, s33                               // 000000007458: D7000904 00004304
	s_wait_alu 0xf1ff                                          // 000000007460: BF88F1FF
	v_add_co_ci_u32_e64 v5, null, 0, v5, s9                    // 000000007464: D5207C05 00260A80
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000746C: BF870003
	v_cmp_le_u64_e32 vcc_lo, s[26:27], v[6:7]                  // 000000007470: 7CB60C1A
	s_or_b32 s40, vcc_lo, s40                                  // 000000007474: 8C28286A
	s_wait_loadcnt 0x0                                         // 000000007478: BFC00000
	v_fmac_f32_e32 v22, v25, v23                               // 00000000747C: 562C2F19
	s_wait_alu 0xfffe                                          // 000000007480: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s40                       // 000000007484: 917E287E
	s_cbranch_execnz 65504                                     // 000000007488: BFA6FFE0 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x80c>
	s_or_b32 exec_lo, exec_lo, s40                             // 00000000748C: 8C7E287E
	s_wait_alu 0xfffe                                          // 000000007490: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s39                             // 000000007494: 8C7E277E
	ds_bpermute_b32 v4, v12, v22                               // 000000007498: DACC0000 0400160C
	s_wait_dscnt 0x0                                           // 0000000074A0: BFC60000
	v_add_f32_e32 v4, v22, v4                                  // 0000000074A4: 06080916
	ds_bpermute_b32 v5, v13, v4                                // 0000000074A8: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 0000000074B0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000074B4: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 0000000074B8: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000074C0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000074C4: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 0000000074C8: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000074D0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000074D4: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 0000000074D8: DACC0000 05000410
	s_and_saveexec_b32 s9, s5                                  // 0000000074E0: BE892005
	s_cbranch_execz 4                                          // 0000000074E4: BFA50004 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x8f8>
	s_wait_dscnt 0x0                                           // 0000000074E8: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 0000000074EC: 06080B04
	ds_store_b32 v9, v4                                        // 0000000074F0: D8340000 00000409
	s_wait_alu 0xfffe                                          // 0000000074F8: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000074FC: 8C7E097E
	s_wait_dscnt 0x0                                           // 000000007500: BFC60000
	s_barrier_signal -1                                        // 000000007504: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007508: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 00000000750C: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s6                                  // 000000007518: BE892006
	s_cbranch_execz 31                                         // 00000000751C: BFA5001F <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x99c>
	v_mov_b32_e32 v4, 0                                        // 000000007520: 7E080280
	s_and_saveexec_b32 s16, s7                                 // 000000007524: BE902007
	ds_load_b32 v4, v10                                        // 000000007528: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000007530: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 000000007534: 8C7E107E
	s_wait_dscnt 0x0                                           // 000000007538: BFC60000
	ds_bpermute_b32 v5, v12, v4                                // 00000000753C: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000007544: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007548: 06080B04
	ds_bpermute_b32 v5, v13, v4                                // 00000000754C: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 000000007554: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007558: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 00000000755C: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 000000007564: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007568: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 00000000756C: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 000000007574: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007578: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 00000000757C: DACC0000 05000410
	s_and_b32 exec_lo, exec_lo, s5                             // 000000007584: 8B7E057E
	s_cbranch_execz 4                                          // 000000007588: BFA50004 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x99c>
	s_wait_dscnt 0x0                                           // 00000000758C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007590: 06080B04
	ds_store_b32 v17, v4                                       // 000000007594: D8340000 00000411
	s_wait_alu 0xfffe                                          // 00000000759C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000075A0: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000075A4: BFC80000
	s_barrier_signal -1                                        // 0000000075A8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000075AC: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000075B0: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s8                                  // 0000000075BC: BE892008
	s_cbranch_execz 65394                                      // 0000000075C0: BFA5FF72 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x78c>
	ds_load_b32 v4, v17                                        // 0000000075C4: D8D80000 04000011
	s_lshl_b32 s16, s10, 2                                     // 0000000075CC: 8410820A
	s_wait_dscnt 0x0                                           // 0000000075D0: BFC60000
	s_wait_kmcnt 0x0                                           // 0000000075D4: BFC70000
	s_wait_alu 0xfffe                                          // 0000000075D8: BF88FFFE
	v_dual_mov_b32 v5, s16 :: v_dual_mul_f32 v4, s31, v4       // 0000000075DC: CA060010 0504081F
	ds_store_b32 v5, v4 offset:1024                            // 0000000075E4: D8340400 00000405
	s_branch 65383                                             // 0000000075EC: BFA0FF67 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x78c>
	v_cmp_gt_u32_e32 vcc_lo, s38, v0                           // 0000000075F0: 7C980026
	v_mov_b32_e32 v4, 0xff7fffff                               // 0000000075F4: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s16, vcc_lo                             // 0000000075FC: BE90206A
	s_cbranch_execz 25                                         // 000000007600: BFA50019 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xa68>
	v_dual_mov_b32 v4, 0xff7fffff :: v_dual_mov_b32 v5, v20    // 000000007604: CA1000FF 04040114 FF7FFFFF
	v_mov_b32_e32 v6, v0                                       // 000000007610: 7E0C0300
	s_mov_b32 s17, 0                                           // 000000007614: BE910080
	ds_load_b32 v7, v5                                         // 000000007618: D8D80000 07000005
	v_add_nc_u32_e32 v6, s30, v6                               // 000000007620: 4A0C0C1E
	v_add_nc_u32_e32 v5, s36, v5                               // 000000007624: 4A0A0A24
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007628: BF870002
	v_cmp_le_u32_e64 s9, s38, v6                               // 00000000762C: D44B0009 00020C26
	s_wait_alu 0xfffe                                          // 000000007634: BF88FFFE
	s_or_b32 s17, s9, s17                                      // 000000007638: 8C111109
	s_wait_dscnt 0x0                                           // 00000000763C: BFC60000
	v_cmp_gt_f32_e64 s10, v7, v4                               // 000000007640: D414000A 00020907
	s_wait_alu 0xf1ff                                          // 000000007648: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000764C: BF870001
	v_cndmask_b32_e64 v4, v4, v7, s10                          // 000000007650: D5010004 002A0F04
	s_wait_alu 0xfffe                                          // 000000007658: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s17                       // 00000000765C: 917E117E
	s_cbranch_execnz 65517                                     // 000000007660: BFA6FFED <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xa18>
	s_or_b32 exec_lo, exec_lo, s17                             // 000000007664: 8C7E117E
	s_wait_alu 0xfffe                                          // 000000007668: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s16                             // 00000000766C: 8C7E107E
	ds_bpermute_b32 v5, v12, v4                                // 000000007670: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000007678: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 00000000767C: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000007684: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000007688: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 00000000768C: D5010004 00260905
	ds_bpermute_b32 v5, v13, v4                                // 000000007694: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 00000000769C: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000076A0: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000076A8: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000076AC: D5010004 00260905
	ds_bpermute_b32 v5, v14, v4                                // 0000000076B4: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000076BC: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000076C0: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000076C8: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000076CC: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000076D0: D5010004 00260905
	ds_bpermute_b32 v5, v15, v4                                // 0000000076D8: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000076E0: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000076E4: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000076EC: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000076F0: D5010004 00260905
	ds_bpermute_b32 v5, v16, v4                                // 0000000076F8: DACC0000 05000410
	s_and_saveexec_b32 s10, s5                                 // 000000007700: BE8A2005
	s_cbranch_execz 9                                          // 000000007704: BFA50009 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xb2c>
	s_wait_dscnt 0x0                                           // 000000007708: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 00000000770C: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000007714: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007718: BF870001
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 00000000771C: D5010004 00260905
	ds_store_b32 v9, v4                                        // 000000007724: D8340000 00000409
	s_wait_alu 0xfffe                                          // 00000000772C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000007730: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007734: BFC80000
	s_barrier_signal -1                                        // 000000007738: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000773C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007740: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s10, s6                                 // 00000000774C: BE8A2006
	s_cbranch_execz 55                                         // 000000007750: BFA50037 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xc30>
	v_mov_b32_e32 v4, 0xff7fffff                               // 000000007754: 7E0802FF FF7FFFFF
	s_and_saveexec_b32 s9, s7                                  // 00000000775C: BE892007
	ds_load_b32 v4, v10                                        // 000000007760: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000007768: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000776C: 8C7E097E
	s_wait_dscnt 0x0                                           // 000000007770: BFC60000
	ds_bpermute_b32 v5, v12, v4                                // 000000007774: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 00000000777C: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000007780: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000007788: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 00000000778C: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000007790: D5010004 00260905
	ds_bpermute_b32 v5, v13, v4                                // 000000007798: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 0000000077A0: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000077A4: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000077AC: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000077B0: D5010004 00260905
	ds_bpermute_b32 v5, v14, v4                                // 0000000077B8: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 0000000077C0: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000077C4: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000077CC: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000077D0: BF8700D1
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000077D4: D5010004 00260905
	ds_bpermute_b32 v5, v15, v4                                // 0000000077DC: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 0000000077E4: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 0000000077E8: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 0000000077F0: BF88F1FF
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 0000000077F4: D5010004 00260905
	ds_bpermute_b32 v5, v16, v4                                // 0000000077FC: DACC0000 05000410
	s_and_b32 exec_lo, exec_lo, s5                             // 000000007804: 8B7E057E
	s_cbranch_execz 9                                          // 000000007808: BFA50009 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xc30>
	s_wait_dscnt 0x0                                           // 00000000780C: BFC60000
	v_cmp_gt_f32_e64 s9, v4, v5                                // 000000007810: D4140009 00020B04
	s_wait_alu 0xf1ff                                          // 000000007818: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000781C: BF870001
	v_cndmask_b32_e64 v4, v5, v4, s9                           // 000000007820: D5010004 00260905
	ds_store_b32 v17, v4                                       // 000000007828: D8340000 00000411
	s_wait_alu 0xfffe                                          // 000000007830: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000007834: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007838: BFC80000
	s_barrier_signal -1                                        // 00000000783C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007840: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007844: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s10, s8                                 // 000000007850: BE8A2008
	s_cbranch_execz 57                                         // 000000007854: BFA50039 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xd3c>
	ds_load_b32 v6, v17                                        // 000000007858: D8D80000 06000011
	s_wait_dscnt 0x0                                           // 000000007860: BFC60000
	v_readfirstlane_b32 s9, v6                                 // 000000007864: 7E120506
	s_cmp_gt_f32 s37, s9                                       // 000000007868: BF440925
	s_cselect_b32 s16, s37, s9                                 // 00000000786C: 98100925
	s_wait_alu 0xfffe                                          // 000000007870: BF88FFFE
	s_sub_f32 s17, s37, s16                                    // 000000007874: A0911025
	v_mov_b32_e32 v5, s16                                      // 000000007878: 7E0A0210
	s_wait_alu 0xfffe                                          // 00000000787C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000007880: BF870529
	s_mul_f32 s9, s17, 0x3fb8aa3b                              // 000000007884: A209FF11 3FB8AA3B
	s_wait_alu 0xfffe                                          // 00000000788C: BF88FFFE
	s_xor_b32 s39, s9, 0x80000000                              // 000000007890: 8D27FF09 80000000
	s_rndne_f32 s40, s9                                        // 000000007898: BEA86309
	s_wait_alu 0xfffe                                          // 00000000789C: BF88FFFE
	s_fmamk_f32 s39, s17, 0x3fb8aa3b, s39                      // 0000000078A0: A3272711 3FB8AA3B
	s_cmp_nlt_f32 s17, 0xc2ce8ed0                              // 0000000078A8: BF4EFF11 C2CE8ED0
	s_sub_f32 s9, s9, s40                                      // 0000000078B0: A0892809
	s_wait_alu 0xfffe                                          // 0000000078B4: BF88FFFE
	s_fmamk_f32 s39, s17, 0x32a5705f, s39                      // 0000000078B8: A3272711 32A5705F
	s_wait_alu 0xfffe                                          // 0000000078C0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000078C4: BF8704BA
	s_add_f32 s9, s9, s39                                      // 0000000078C8: A0092709
	s_cvt_i32_f32 s39, s40                                     // 0000000078CC: BEA76628
	s_wait_alu 0xfffe                                          // 0000000078D0: BF88FFFE
	v_s_exp_f32 s9, s9                                         // 0000000078D4: D6800009 00000009
	s_wait_alu 0xf1ff                                          // 0000000078DC: BF88F1FF
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000078E0: BF8700C5
	v_ldexp_f32 v4, s9, s39                                    // 0000000078E4: D71C0004 00004E09
	s_cselect_b32 s9, -1, 0                                    // 0000000078EC: 980980C1
	s_cmp_ngt_f32 s17, 0x42b17218                              // 0000000078F0: BF4BFF11 42B17218
	s_wait_alu 0xfffe                                          // 0000000078F8: BF88FFFE
	v_cndmask_b32_e64 v4, 0, v4, s9                            // 0000000078FC: D5010004 00260880
	s_cselect_b32 s9, -1, 0                                    // 000000007904: 980980C1
	s_cmp_nle_f32 s37, 0xff61b1e6                              // 000000007908: BF4CFF25 FF61B1E6
	s_wait_alu 0xfffe                                          // 000000007910: BF88FFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000007914: BF8700B1
	v_cndmask_b32_e64 v4, 0x7f800000, v4, s9                   // 000000007918: D5010004 002608FF 7F800000
	s_cselect_b32 s9, -1, 0                                    // 000000007924: 980980C1
	s_wait_alu 0xfffe                                          // 000000007928: BF88FFFE
	v_cndmask_b32_e64 v4, 0, v4, s9                            // 00000000792C: D5010004 00260880
	ds_store_b96 v17, v[4:6] offset:1280                       // 000000007934: DB780500 00000411
	s_wait_alu 0xfffe                                          // 00000000793C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000007940: 8C7E0A7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007944: BFC80000
	s_barrier_signal -1                                        // 000000007948: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000794C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007950: EE0AC07C 00040000 00000000
	ds_load_b32 v4, v17 offset:1284                            // 00000000795C: D8D80504 04000011
	s_wait_dscnt 0x0                                           // 000000007964: BFC60000
	v_readfirstlane_b32 s37, v4                                // 000000007968: 7E4A0504
	v_mov_b32_e32 v4, 0                                        // 00000000796C: 7E080280
	s_and_saveexec_b32 s9, vcc_lo                              // 000000007970: BE89206A
	s_cbranch_execz 49                                         // 000000007974: BFA50031 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xe3c>
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, v20             // 000000007978: CA100080 04040114
	v_mov_b32_e32 v6, v0                                       // 000000007980: 7E0C0300
	s_mov_b32 s10, 0                                           // 000000007984: BE8A0080
	ds_load_b32 v7, v5                                         // 000000007988: D8D80000 07000005
	s_wait_dscnt 0x0                                           // 000000007990: BFC60000
	s_wait_alu 0xf1ff                                          // 000000007994: BF88F1FF
	v_dual_subrev_f32 v7, s37, v7 :: v_dual_add_nc_u32 v6, s30, v6// 000000007998: C9A00E25 07060C1E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000079A0: BF870091
	v_mul_f32_e32 v22, 0x3fb8aa3b, v7                          // 0000000079A4: 102C0EFF 3FB8AA3B
	v_fma_f32 v23, 0x3fb8aa3b, v7, -v22                        // 0000000079AC: D6130017 845A0EFF 3FB8AA3B
	v_rndne_f32_e32 v24, v22                                   // 0000000079B8: 7E304716
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000079BC: BF870221
	v_sub_f32_e32 v22, v22, v24                                // 0000000079C0: 082C3116
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v7                   // 0000000079C4: 7C360EFF C2CE8ED0
	v_fmac_f32_e32 v23, 0x32a5705f, v7                         // 0000000079CC: 562E0EFF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000079D4: BF870121
	v_add_f32_e32 v22, v22, v23                                // 0000000079D8: 062C2F16
	v_cvt_i32_f32_e32 v23, v24                                 // 0000000079DC: 7E2E1118
	v_exp_f32_e32 v22, v22                                     // 0000000079E0: 7E2C4B16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000079E4: BF8700A5
	v_ldexp_f32 v22, v22, v23                                  // 0000000079E8: D71C0016 00022F16
	s_wait_alu 0xfffd                                          // 0000000079F0: BF88FFFD
	v_cndmask_b32_e32 v22, 0, v22, vcc_lo                      // 0000000079F4: 022C2C80
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v7                   // 0000000079F8: 7C3C0EFF 42B17218
	s_wait_alu 0xfffd                                          // 000000007A00: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007A04: BF870002
	v_cndmask_b32_e32 v7, 0x7f800000, v22, vcc_lo              // 000000007A08: 020E2CFF 7F800000
	v_cmp_le_u32_e32 vcc_lo, s38, v6                           // 000000007A10: 7C960C26
	ds_store_b32 v5, v7                                        // 000000007A14: D8340000 00000705
	v_dual_add_f32 v4, v4, v7 :: v_dual_add_nc_u32 v5, s36, v5 // 000000007A1C: C9200F04 04040A24
	s_wait_alu 0xfffe                                          // 000000007A24: BF88FFFE
	s_or_b32 s10, vcc_lo, s10                                  // 000000007A28: 8C0A0A6A
	s_wait_alu 0xfffe                                          // 000000007A2C: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s10                       // 000000007A30: 917E0A7E
	s_cbranch_execnz 65492                                     // 000000007A34: BFA6FFD4 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xd88>
	s_or_b32 exec_lo, exec_lo, s10                             // 000000007A38: 8C7E0A7E
	s_wait_alu 0xfffe                                          // 000000007A3C: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007A40: 8C7E097E
	ds_bpermute_b32 v5, v12, v4                                // 000000007A44: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000007A4C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007A50: 06080B04
	ds_bpermute_b32 v5, v13, v4                                // 000000007A54: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 000000007A5C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007A60: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 000000007A64: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 000000007A6C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007A70: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 000000007A74: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 000000007A7C: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007A80: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 000000007A84: DACC0000 05000410
	s_and_saveexec_b32 s9, s5                                  // 000000007A8C: BE892005
	s_cbranch_execz 4                                          // 000000007A90: BFA50004 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xea4>
	s_wait_dscnt 0x0                                           // 000000007A94: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007A98: 06080B04
	ds_store_b32 v9, v4                                        // 000000007A9C: D8340000 00000409
	s_wait_alu 0xfffe                                          // 000000007AA4: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007AA8: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007AAC: BFC80000
	s_barrier_signal -1                                        // 000000007AB0: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007AB4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007AB8: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s6                                  // 000000007AC4: BE892006
	s_cbranch_execz 31                                         // 000000007AC8: BFA5001F <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xf48>
	v_mov_b32_e32 v4, 0                                        // 000000007ACC: 7E080280
	s_and_saveexec_b32 s10, s7                                 // 000000007AD0: BE8A2007
	ds_load_b32 v4, v10                                        // 000000007AD4: D8D80000 0400000A
	s_wait_alu 0xfffe                                          // 000000007ADC: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s10                             // 000000007AE0: 8C7E0A7E
	s_wait_dscnt 0x0                                           // 000000007AE4: BFC60000
	ds_bpermute_b32 v5, v12, v4                                // 000000007AE8: DACC0000 0500040C
	s_wait_dscnt 0x0                                           // 000000007AF0: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007AF4: 06080B04
	ds_bpermute_b32 v5, v13, v4                                // 000000007AF8: DACC0000 0500040D
	s_wait_dscnt 0x0                                           // 000000007B00: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007B04: 06080B04
	ds_bpermute_b32 v5, v14, v4                                // 000000007B08: DACC0000 0500040E
	s_wait_dscnt 0x0                                           // 000000007B10: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007B14: 06080B04
	ds_bpermute_b32 v5, v15, v4                                // 000000007B18: DACC0000 0500040F
	s_wait_dscnt 0x0                                           // 000000007B20: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007B24: 06080B04
	ds_bpermute_b32 v5, v16, v4                                // 000000007B28: DACC0000 05000410
	s_and_b32 exec_lo, exec_lo, s5                             // 000000007B30: 8B7E057E
	s_cbranch_execz 4                                          // 000000007B34: BFA50004 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xf48>
	s_wait_dscnt 0x0                                           // 000000007B38: BFC60000
	v_add_f32_e32 v4, v4, v5                                   // 000000007B3C: 06080B04
	ds_store_b32 v17, v4                                       // 000000007B40: D8340000 00000411
	s_wait_alu 0xfffe                                          // 000000007B48: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007B4C: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007B50: BFC80000
	s_barrier_signal -1                                        // 000000007B54: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007B58: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007B5C: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s8                                  // 000000007B68: BE892008
	s_cbranch_execz 5                                          // 000000007B6C: BFA50005 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xf84>
	ds_load_b32 v4, v17                                        // 000000007B70: D8D80000 04000011
	s_wait_dscnt 0x0                                           // 000000007B78: BFC60000
	ds_store_b32 v17, v4 offset:1292                           // 000000007B7C: D834050C 00000411
	s_wait_alu 0xfffe                                          // 000000007B84: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007B88: 8C7E097E
	s_wait_loadcnt_dscnt 0x0                                   // 000000007B8C: BFC80000
	s_barrier_signal -1                                        // 000000007B90: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007B94: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007B98: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s9, s3                                  // 000000007BA4: BE892003
	s_wait_alu 0xfffe                                          // 000000007BA8: BF88FFFE
	s_xor_b32 s9, exec_lo, s9                                  // 000000007BAC: 8D09097E
	ds_load_b32 v5, v17 offset:1280                            // 000000007BB0: D8D80500 05000011
	s_wait_alu 0xfffe                                          // 000000007BB8: BF88FFFE
	s_and_not1_saveexec_b32 s9, s9                             // 000000007BBC: BE893009
	s_cbranch_execz 153                                        // 000000007BC0: BFA50099 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x1228>
	v_cmp_lt_u64_e64 s10, s[12:13], 4                          // 000000007BC4: D459000A 0001080C
	v_mov_b32_e32 v4, 0                                        // 000000007BCC: 7E080280
	s_max_u32 s13, s38, 1                                      // 000000007BD0: 8A8D8126
	s_and_b32 vcc_lo, exec_lo, s10                             // 000000007BD4: 8B6A0A7E
	s_wait_alu 0xfffe                                          // 000000007BD8: BF88FFFE
	s_cbranch_vccnz 99                                         // 000000007BDC: BFA40063 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x116c>
	s_and_b32 s12, s13, 0x7c                                   // 000000007BE0: 8B0CFF0D 0000007C
	s_mov_b32 s10, 0                                           // 000000007BE8: BE8A0080
	s_movk_i32 s16, 0x400                                      // 000000007BEC: B0100400
	s_wait_alu 0xfffe                                          // 000000007BF0: BF88FFFE
	s_add_nc_u64 s[38:39], s[34:35], s[10:11]                  // 000000007BF4: A9A60A22
	s_or_b32 s40, s10, 1                                       // 000000007BF8: 8C28810A
	s_mov_b32 s41, s11                                         // 000000007BFC: BEA9000B
	s_wait_alu 0xfffe                                          // 000000007C00: BF88FFFE
	s_mul_u64 s[38:39], s[38:39], s[24:25]                     // 000000007C04: AAA61826
	s_add_nc_u64 s[40:41], s[34:35], s[40:41]                  // 000000007C08: A9A82822
	s_wait_alu 0xfffe                                          // 000000007C0C: BF88FFFE
	s_add_nc_u64 s[38:39], s[38:39], s[22:23]                  // 000000007C10: A9A61626
	s_mul_u64 s[40:41], s[40:41], s[24:25]                     // 000000007C14: AAA81828
	s_wait_alu 0xfffe                                          // 000000007C18: BF88FFFE
	s_mul_u64 s[38:39], s[38:39], s[20:21]                     // 000000007C1C: AAA61426
	s_add_nc_u64 s[40:41], s[40:41], s[22:23]                  // 000000007C20: A9A81628
	s_wait_alu 0xfffe                                          // 000000007C24: BF88FFFE
	s_lshl_b64 s[38:39], s[38:39], 2                           // 000000007C28: 84A68226
	s_or_b32 s42, s10, 2                                       // 000000007C2C: 8C2A820A
	s_mov_b32 s43, s11                                         // 000000007C30: BEAB000B
	s_mul_u64 s[40:41], s[40:41], s[20:21]                     // 000000007C34: AAA81428
	s_wait_dscnt 0x0                                           // 000000007C38: BFC60000
	s_wait_alu 0xfffe                                          // 000000007C3C: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v18, s38                          // 000000007C40: D7006A05 00004D12
	s_or_b32 s44, s10, 3                                       // 000000007C48: 8C2C830A
	s_mov_b32 s45, s11                                         // 000000007C4C: BEAD000B
	s_add_nc_u64 s[42:43], s[34:35], s[42:43]                  // 000000007C50: A9AA2A22
	s_wait_alu 0xfffd                                          // 000000007C54: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s39, v19, vcc_lo             // 000000007C58: D5207C06 01AA2627
	s_lshl_b64 s[38:39], s[40:41], 2                           // 000000007C60: 84A68228
	s_add_nc_u64 s[44:45], s[34:35], s[44:45]                  // 000000007C64: A9AC2C22
	s_wait_alu 0xfffe                                          // 000000007C68: BF88FFFE
	s_mul_u64 s[42:43], s[42:43], s[24:25]                     // 000000007C6C: AAAA182A
	v_add_co_u32 v22, vcc_lo, v18, s38                         // 000000007C70: D7006A16 00004D12
	s_mul_u64 s[44:45], s[44:45], s[24:25]                     // 000000007C78: AAAC182C
	s_wait_alu 0xfffe                                          // 000000007C7C: BF88FFFE
	s_add_nc_u64 s[42:43], s[42:43], s[22:23]                  // 000000007C80: A9AA162A
	s_wait_alu 0xfffd                                          // 000000007C84: BF88FFFD
	v_add_co_ci_u32_e64 v23, null, s39, v19, vcc_lo            // 000000007C88: D5207C17 01AA2627
	s_add_nc_u64 s[44:45], s[44:45], s[22:23]                  // 000000007C90: A9AC162C
	s_wait_alu 0xfffe                                          // 000000007C94: BF88FFFE
	s_mul_u64 s[42:43], s[42:43], s[20:21]                     // 000000007C98: AAAA142A
	s_clause 0x1                                               // 000000007C9C: BF850001
	global_load_b32 v7, v[5:6], off                            // 000000007CA0: EE05007C 00000007 00000005
	global_load_b32 v26, v[22:23], off                         // 000000007CAC: EE05007C 0000001A 00000016
	s_mul_u64 s[44:45], s[44:45], s[20:21]                     // 000000007CB8: AAAC142C
	s_wait_alu 0xfffe                                          // 000000007CBC: BF88FFFE
	s_lshl_b64 s[40:41], s[42:43], 2                           // 000000007CC0: 84A8822A
	s_lshl_b64 s[42:43], s[44:45], 2                           // 000000007CC4: 84AA822C
	s_wait_alu 0xfffe                                          // 000000007CC8: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v18, s40                          // 000000007CCC: D7006A05 00005112
	s_wait_alu 0xfffd                                          // 000000007CD4: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s41, v19, vcc_lo             // 000000007CD8: D5207C06 01AA2629
	v_add_co_u32 v22, vcc_lo, v18, s42                         // 000000007CE0: D7006A16 00005512
	s_wait_alu 0xfffd                                          // 000000007CE8: BF88FFFD
	v_add_co_ci_u32_e64 v23, null, s43, v19, vcc_lo            // 000000007CEC: D5207C17 01AA262B
	s_clause 0x1                                               // 000000007CF4: BF850001
	global_load_b32 v5, v[5:6], off                            // 000000007CF8: EE05007C 00000005 00000005
	global_load_b32 v6, v[22:23], off                          // 000000007D04: EE05007C 00000006 00000016
	v_mov_b32_e32 v22, s16                                     // 000000007D10: 7E2C0210
	s_add_co_i32 s10, s10, 4                                   // 000000007D14: 810A840A
	s_add_co_i32 s16, s16, 16                                  // 000000007D18: 81109010
	s_wait_alu 0xfffe                                          // 000000007D1C: BF88FFFE
	s_cmp_eq_u32 s12, s10                                      // 000000007D20: BF060A0C
	ds_load_b128 v[22:25], v22                                 // 000000007D24: DBFC0000 16000016
	s_wait_loadcnt_dscnt 0x300                                 // 000000007D2C: BFC80300
	v_fmac_f32_e32 v4, v22, v7                                 // 000000007D30: 56080F16
	s_wait_loadcnt 0x2                                         // 000000007D34: BFC00002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000007D38: BF8700A1
	v_fmac_f32_e32 v4, v23, v26                                // 000000007D3C: 56083517
	s_wait_loadcnt 0x1                                         // 000000007D40: BFC00001
	v_fmac_f32_e32 v4, v24, v5                                 // 000000007D44: 56080B18
	s_wait_loadcnt 0x0                                         // 000000007D48: BFC00000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007D4C: BF870001
	v_fmac_f32_e32 v4, v25, v6                                 // 000000007D50: 56080D19
	s_cbranch_scc0 65446                                       // 000000007D54: BFA1FFA6 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0xff0>
	s_and_b32 s13, s13, 3                                      // 000000007D58: 8B0D830D
	s_wait_alu 0xfffe                                          // 000000007D5C: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 000000007D60: BF06800D
	s_cbranch_scc0 6                                           // 000000007D64: BFA10006 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x1180>
	s_branch 40                                                // 000000007D68: BFA00028 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x120c>
	s_mov_b32 s12, 0                                           // 000000007D6C: BE8C0080
	s_and_b32 s13, s13, 3                                      // 000000007D70: 8B0D830D
	s_wait_alu 0xfffe                                          // 000000007D74: BF88FFFE
	s_cmp_eq_u32 s13, 0                                        // 000000007D78: BF06800D
	s_cbranch_scc1 35                                          // 000000007D7C: BFA20023 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x120c>
	s_lshl_b32 s10, s12, 2                                     // 000000007D80: 840A820C
	s_wait_alu 0xfffe                                          // 000000007D84: BF88FFFE
	s_or_b32 s16, s10, 0x400                                   // 000000007D88: 8C10FF0A 00000400
	s_mov_b32 s10, s12                                         // 000000007D90: BE8A000C
	s_wait_alu 0xfffe                                          // 000000007D94: BF88FFFE
	s_add_nc_u64 s[38:39], s[34:35], s[10:11]                  // 000000007D98: A9A60A22
	s_add_co_i32 s13, s13, -1                                  // 000000007D9C: 810DC10D
	s_wait_alu 0xfffe                                          // 000000007DA0: BF88FFFE
	s_mul_u64 s[38:39], s[38:39], s[24:25]                     // 000000007DA4: AAA61826
	s_add_co_i32 s10, s10, 1                                   // 000000007DA8: 810A810A
	s_wait_alu 0xfffe                                          // 000000007DAC: BF88FFFE
	s_add_nc_u64 s[38:39], s[38:39], s[22:23]                  // 000000007DB0: A9A61626
	s_wait_alu 0xfffe                                          // 000000007DB4: BF88FFFE
	s_mul_u64 s[38:39], s[38:39], s[20:21]                     // 000000007DB8: AAA61426
	s_wait_alu 0xfffe                                          // 000000007DBC: BF88FFFE
	s_lshl_b64 s[38:39], s[38:39], 2                           // 000000007DC0: 84A68226
	s_wait_dscnt 0x0                                           // 000000007DC4: BFC60000
	s_wait_alu 0xfffe                                          // 000000007DC8: BF88FFFE
	v_add_co_u32 v5, vcc_lo, v18, s38                          // 000000007DCC: D7006A05 00004D12
	s_wait_alu 0xfffd                                          // 000000007DD4: BF88FFFD
	v_add_co_ci_u32_e64 v6, null, s39, v19, vcc_lo             // 000000007DD8: D5207C06 01AA2627
	global_load_b32 v5, v[5:6], off                            // 000000007DE0: EE05007C 00000005 00000005
	v_mov_b32_e32 v6, s16                                      // 000000007DEC: 7E0C0210
	s_add_co_i32 s16, s16, 4                                   // 000000007DF0: 81108410
	s_cmp_lg_u32 s13, 0                                        // 000000007DF4: BF07800D
	ds_load_b32 v6, v6                                         // 000000007DF8: D8D80000 06000006
	s_wait_loadcnt_dscnt 0x0                                   // 000000007E00: BFC80000
	v_fmac_f32_e32 v4, v6, v5                                  // 000000007E04: 56080B06
	s_cbranch_scc1 65506                                       // 000000007E08: BFA2FFE2 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x1194>
	s_wait_dscnt 0x0                                           // 000000007E0C: BFC60000
	ds_load_b32 v5, v17 offset:1280                            // 000000007E10: D8D80500 05000011
	s_wait_dscnt 0x0                                           // 000000007E18: BFC60000
	v_fmac_f32_e32 v4, v11, v5                                 // 000000007E1C: 56080B0B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007E20: BF870001
	v_mov_b32_e32 v11, v4                                      // 000000007E24: 7E160304
	s_wait_alu 0xfffe                                          // 000000007E28: BF88FFFE
	s_or_b32 exec_lo, exec_lo, s9                              // 000000007E2C: 8C7E097E
	ds_load_b32 v4, v17 offset:1292                            // 000000007E30: D8D8050C 04000011
	s_add_nc_u64 s[34:35], s[34:35], 64                        // 000000007E38: A9A2C022
	s_wait_loadcnt_dscnt 0x0                                   // 000000007E3C: BFC80000
	s_wait_alu 0xfffe                                          // 000000007E40: BF88FFFE
	v_cmp_ge_u64_e64 s9, s[34:35], s[18:19]                    // 000000007E44: D45E0009 00002422
	s_barrier_signal -1                                        // 000000007E4C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000007E50: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007E54: EE0AC07C 00040000 00000000
	s_and_b32 vcc_lo, exec_lo, s9                              // 000000007E60: 8B6A097E
	v_fmac_f32_e32 v4, v21, v5                                 // 000000007E64: 56080B15
	s_wait_alu 0xfffe                                          // 000000007E68: BF88FFFE
	s_cbranch_vccnz 3                                          // 000000007E6C: BFA40003 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x127c>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007E70: BF870001
	v_mov_b32_e32 v21, v4                                      // 000000007E74: 7E2A0304
	s_branch 64828                                             // 000000007E78: BFA0FD3C <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x76c>
	s_and_saveexec_b32 s3, s2                                  // 000000007E7C: BE832002
	s_cbranch_execz 34                                         // 000000007E80: BFA50022 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x130c>
	v_div_scale_f32 v0, null, v4, v4, v11                      // 000000007E84: D6FC7C00 042E0904
	s_load_b64 s[0:1], s[0:1], 0x50                            // 000000007E8C: F4002000 F8000050
	s_mul_u64 s[2:3], s[20:21], s[28:29]                       // 000000007E94: AA821C14
	s_wait_alu 0xfffe                                          // 000000007E98: BF88FFFE
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000007E9C: 84828202
	v_rcp_f32_e32 v1, v0                                       // 000000007EA0: 7E025500
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007EA4: BF870095
	v_fma_f32 v2, -v0, v1, 1.0                                 // 000000007EA8: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 000000007EB0: 56020302
	v_div_scale_f32 v2, vcc_lo, v11, v4, v11                   // 000000007EB4: D6FC6A02 042E090B
	s_wait_kmcnt 0x0                                           // 000000007EBC: BFC70000
	s_wait_alu 0xfffe                                          // 000000007EC0: BF88FFFE
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000007EC4: A9800200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007EC8: BF870091
	v_mul_f32_e32 v3, v2, v1                                   // 000000007ECC: 10060302
	v_fma_f32 v5, -v0, v3, v2                                  // 000000007ED0: D6130005 240A0700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007ED8: BF870091
	v_fmac_f32_e32 v3, v5, v1                                  // 000000007EDC: 56060305
	v_fma_f32 v0, -v0, v3, v2                                  // 000000007EE0: D6130000 240A0700
	s_wait_alu 0xfffd                                          // 000000007EE8: BF88FFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007EEC: BF870091
	v_div_fmas_f32 v0, v0, v1, v3                              // 000000007EF0: D6370000 040E0300
	v_div_fixup_f32 v0, v0, v4, v11                            // 000000007EF8: D6270000 042E0900
	global_store_b32 v8, v0, s[0:1]                            // 000000007F00: EE068000 00000000 00000008
	s_endpgm                                                   // 000000007F0C: BFB00000
	s_branch 64509                                             // 000000007F10: BFA0FBFD <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x308>
	s_branch 64676                                             // 000000007F14: BFA0FCA4 <ullm_sq8_0_flash2_staged_wave32_prototype_kernel+0x5a8>
	s_nop 0                                                    // 000000007F18: BF800000
	s_nop 0                                                    // 000000007F1C: BF800000
	s_nop 0                                                    // 000000007F20: BF800000
	s_nop 0                                                    // 000000007F24: BF800000
	s_nop 0                                                    // 000000007F28: BF800000
	s_nop 0                                                    // 000000007F2C: BF800000
	s_nop 0                                                    // 000000007F30: BF800000
	s_nop 0                                                    // 000000007F34: BF800000
	s_nop 0                                                    // 000000007F38: BF800000
	s_nop 0                                                    // 000000007F3C: BF800000
	s_nop 0                                                    // 000000007F40: BF800000
	s_nop 0                                                    // 000000007F44: BF800000
	s_nop 0                                                    // 000000007F48: BF800000
	s_nop 0                                                    // 000000007F4C: BF800000
	s_nop 0                                                    // 000000007F50: BF800000
	s_nop 0                                                    // 000000007F54: BF800000
	s_nop 0                                                    // 000000007F58: BF800000
	s_nop 0                                                    // 000000007F5C: BF800000
	s_nop 0                                                    // 000000007F60: BF800000
	s_nop 0                                                    // 000000007F64: BF800000
	s_nop 0                                                    // 000000007F68: BF800000
	s_nop 0                                                    // 000000007F6C: BF800000
	s_nop 0                                                    // 000000007F70: BF800000
	s_nop 0                                                    // 000000007F74: BF800000
	s_nop 0                                                    // 000000007F78: BF800000
	s_nop 0                                                    // 000000007F7C: BF800000
	s_nop 0                                                    // 000000007F80: BF800000
	s_nop 0                                                    // 000000007F84: BF800000
	s_nop 0                                                    // 000000007F88: BF800000
	s_nop 0                                                    // 000000007F8C: BF800000
	s_nop 0                                                    // 000000007F90: BF800000
	s_nop 0                                                    // 000000007F94: BF800000
	s_nop 0                                                    // 000000007F98: BF800000
	s_nop 0                                                    // 000000007F9C: BF800000
	s_nop 0                                                    // 000000007FA0: BF800000
	s_nop 0                                                    // 000000007FA4: BF800000
	s_nop 0                                                    // 000000007FA8: BF800000
	s_nop 0                                                    // 000000007FAC: BF800000
	s_nop 0                                                    // 000000007FB0: BF800000
	s_nop 0                                                    // 000000007FB4: BF800000
	s_nop 0                                                    // 000000007FB8: BF800000
	s_nop 0                                                    // 000000007FBC: BF800000
	s_nop 0                                                    // 000000007FC0: BF800000
	s_nop 0                                                    // 000000007FC4: BF800000
	s_nop 0                                                    // 000000007FC8: BF800000
	s_nop 0                                                    // 000000007FCC: BF800000
	s_nop 0                                                    // 000000007FD0: BF800000
	s_nop 0                                                    // 000000007FD4: BF800000
	s_nop 0                                                    // 000000007FD8: BF800000
	s_nop 0                                                    // 000000007FDC: BF800000
	s_nop 0                                                    // 000000007FE0: BF800000
	s_nop 0                                                    // 000000007FE4: BF800000
	s_nop 0                                                    // 000000007FE8: BF800000
	s_nop 0                                                    // 000000007FEC: BF800000
	s_nop 0                                                    // 000000007FF0: BF800000
	s_nop 0                                                    // 000000007FF4: BF800000
	s_nop 0                                                    // 000000007FF8: BF800000
	s_nop 0                                                    // 000000007FFC: BF800000

0000000000008000 <ullm_sq8_0_pmc_probe_kernel>:
	s_clause 0x1                                               // 000000008000: BF850001
	s_load_b32 s2, s[0:1], 0x24                                // 000000008004: F4000080 F8000024
	s_load_b64 s[4:5], s[0:1], 0x10                            // 00000000800C: F4002100 F8000010
	v_mov_b32_e32 v1, 0                                        // 000000008014: 7E020280
	s_wait_kmcnt 0x0                                           // 000000008018: BFC70000
	s_and_b32 s6, s2, 0xffff                                   // 00000000801C: 8B06FF02 0000FFFF
	s_mov_b32 s2, exec_lo                                      // 000000008024: BE82007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008028: BF870091
	v_mad_co_u64_u32 v[2:3], null, s6, ttmp9, v[0:1]           // 00000000802C: D6FE7C02 0400EA06
	v_cmpx_gt_u64_e64 s[4:5], v[2:3]                           // 000000008034: D4DC007E 00020404
	s_cbranch_execz 61                                         // 00000000803C: BFA5003D <ullm_sq8_0_pmc_probe_kernel+0x134>
	s_add_nc_u64 s[2:3], s[0:1], 24                            // 000000008040: A9829800
	v_lshlrev_b64_e32 v[4:5], 2, v[2:3]                        // 000000008044: 3E080482
	s_load_b32 s8, s[2:3], 0x0                                 // 000000008048: F4000201 F8000000
	s_load_b128 s[0:3], s[0:1], 0x0                            // 000000008050: F4004000 F8000000
	s_mov_b32 s7, 0                                            // 000000008058: BE870080
	s_wait_alu 0xfffe                                          // 00000000805C: BF88FFFE
	s_mov_b32 s9, s7                                           // 000000008060: BE890007
	s_wait_kmcnt 0x0                                           // 000000008064: BFC70000
	s_mul_u64 s[8:9], s[6:7], s[8:9]                           // 000000008068: AA880806
	v_add_co_u32 v6, vcc_lo, s0, v4                            // 00000000806C: D7006A06 00020800
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008074: BF870001
	v_add_co_ci_u32_e64 v7, null, s1, v5, vcc_lo               // 000000008078: D5207C07 01AA0A01
	s_lshl_b64 s[10:11], s[8:9], 2                             // 000000008080: 848A8208
	flat_load_b32 v0, v[6:7] scope:SCOPE_SYS                   // 000000008084: EC05007C 000C0000 00000006
	s_wait_loadcnt 0x0                                         // 000000008090: BFC00000
	v_add_co_u32 v2, vcc_lo, v2, s8                            // 000000008094: D7006A02 00001102
	s_wait_alu 0xfffd                                          // 00000000809C: BF88FFFD
	v_add_co_ci_u32_e64 v3, null, s9, v3, vcc_lo               // 0000000080A0: D5207C03 01AA0609
	v_add_co_u32 v6, s0, v6, s10                               // 0000000080A8: D7000006 00001506
	s_wait_alu 0xf1ff                                          // 0000000080B0: BF88F1FF
	v_add_co_ci_u32_e64 v7, null, s11, v7, s0                  // 0000000080B4: D5207C07 00020E0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 0000000080BC: BF870153
	v_cmp_le_u64_e32 vcc_lo, s[4:5], v[2:3]                    // 0000000080C0: 7CB60404
	s_or_b32 s7, vcc_lo, s7                                    // 0000000080C4: 8C07076A
	s_wait_dscnt 0x0                                           // 0000000080C8: BFC60000
	v_dual_fmamk_f32 v1, v1, 0x3f800408, v0 :: v_dual_mul_f32 v8, 0.5, v0// 0000000080CC: C8860101 010800F0 3F800408
	v_mul_f32_e32 v9, 0x3e800000, v0                           // 0000000080D8: 101200FF 3E800000
	v_fmac_f32_e32 v8, 0x3f7ff7f0, v1                          // 0000000080E0: 561002FF 3F7FF7F0
	v_mul_f32_e32 v1, 0x3e000000, v0                           // 0000000080E8: 100200FF 3E000000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000080F0: BF870092
	v_fmac_f32_e32 v9, 0x3f800104, v8                          // 0000000080F4: 561210FF 3F800104
	v_fmac_f32_e32 v1, 0x3f7ffdf8, v9                          // 0000000080FC: 560212FF 3F7FFDF8
	s_wait_alu 0xfffe                                          // 000000008104: BF88FFFE
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000008108: 917E077E
	s_cbranch_execnz 65501                                     // 00000000810C: BFA6FFDD <ullm_sq8_0_pmc_probe_kernel+0x84>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000008110: 8C7E077E
	v_add_co_u32 v2, vcc_lo, s2, v4                            // 000000008114: D7006A02 00020802
	s_wait_alu 0xfffd                                          // 00000000811C: BF88FFFD
	v_add_co_ci_u32_e64 v3, null, s3, v5, vcc_lo               // 000000008120: D5207C03 01AA0A03
	global_store_b32 v[2:3], v1, off                           // 000000008128: EE06807C 00800000 00000002
	s_endpgm                                                   // 000000008134: BFB00000
	s_code_end                                                 // 000000008138: BF9F0000
	s_code_end                                                 // 00000000813C: BF9F0000
	s_code_end                                                 // 000000008140: BF9F0000
	s_code_end                                                 // 000000008144: BF9F0000
	s_code_end                                                 // 000000008148: BF9F0000
	s_code_end                                                 // 00000000814C: BF9F0000
	s_code_end                                                 // 000000008150: BF9F0000
	s_code_end                                                 // 000000008154: BF9F0000
	s_code_end                                                 // 000000008158: BF9F0000
	s_code_end                                                 // 00000000815C: BF9F0000
	s_code_end                                                 // 000000008160: BF9F0000
	s_code_end                                                 // 000000008164: BF9F0000
	s_code_end                                                 // 000000008168: BF9F0000
	s_code_end                                                 // 00000000816C: BF9F0000
	s_code_end                                                 // 000000008170: BF9F0000
	s_code_end                                                 // 000000008174: BF9F0000
	s_code_end                                                 // 000000008178: BF9F0000
	s_code_end                                                 // 00000000817C: BF9F0000
	s_code_end                                                 // 000000008180: BF9F0000
	s_code_end                                                 // 000000008184: BF9F0000
	s_code_end                                                 // 000000008188: BF9F0000
	s_code_end                                                 // 00000000818C: BF9F0000
	s_code_end                                                 // 000000008190: BF9F0000
	s_code_end                                                 // 000000008194: BF9F0000
	s_code_end                                                 // 000000008198: BF9F0000
	s_code_end                                                 // 00000000819C: BF9F0000
	s_code_end                                                 // 0000000081A0: BF9F0000
	s_code_end                                                 // 0000000081A4: BF9F0000
	s_code_end                                                 // 0000000081A8: BF9F0000
	s_code_end                                                 // 0000000081AC: BF9F0000
	s_code_end                                                 // 0000000081B0: BF9F0000
	s_code_end                                                 // 0000000081B4: BF9F0000
	s_code_end                                                 // 0000000081B8: BF9F0000
	s_code_end                                                 // 0000000081BC: BF9F0000
	s_code_end                                                 // 0000000081C0: BF9F0000
	s_code_end                                                 // 0000000081C4: BF9F0000
	s_code_end                                                 // 0000000081C8: BF9F0000
	s_code_end                                                 // 0000000081CC: BF9F0000
	s_code_end                                                 // 0000000081D0: BF9F0000
	s_code_end                                                 // 0000000081D4: BF9F0000
	s_code_end                                                 // 0000000081D8: BF9F0000
	s_code_end                                                 // 0000000081DC: BF9F0000
	s_code_end                                                 // 0000000081E0: BF9F0000
	s_code_end                                                 // 0000000081E4: BF9F0000
	s_code_end                                                 // 0000000081E8: BF9F0000
	s_code_end                                                 // 0000000081EC: BF9F0000
	s_code_end                                                 // 0000000081F0: BF9F0000
	s_code_end                                                 // 0000000081F4: BF9F0000
	s_code_end                                                 // 0000000081F8: BF9F0000
	s_code_end                                                 // 0000000081FC: BF9F0000
	s_code_end                                                 // 000000008200: BF9F0000
	s_code_end                                                 // 000000008204: BF9F0000
	s_code_end                                                 // 000000008208: BF9F0000
	s_code_end                                                 // 00000000820C: BF9F0000
	s_code_end                                                 // 000000008210: BF9F0000
	s_code_end                                                 // 000000008214: BF9F0000
	s_code_end                                                 // 000000008218: BF9F0000
	s_code_end                                                 // 00000000821C: BF9F0000
	s_code_end                                                 // 000000008220: BF9F0000
	s_code_end                                                 // 000000008224: BF9F0000
	s_code_end                                                 // 000000008228: BF9F0000
	s_code_end                                                 // 00000000822C: BF9F0000
	s_code_end                                                 // 000000008230: BF9F0000
	s_code_end                                                 // 000000008234: BF9F0000
	s_code_end                                                 // 000000008238: BF9F0000
	s_code_end                                                 // 00000000823C: BF9F0000
	s_code_end                                                 // 000000008240: BF9F0000
	s_code_end                                                 // 000000008244: BF9F0000
	s_code_end                                                 // 000000008248: BF9F0000
	s_code_end                                                 // 00000000824C: BF9F0000
	s_code_end                                                 // 000000008250: BF9F0000
	s_code_end                                                 // 000000008254: BF9F0000
	s_code_end                                                 // 000000008258: BF9F0000
	s_code_end                                                 // 00000000825C: BF9F0000
	s_code_end                                                 // 000000008260: BF9F0000
	s_code_end                                                 // 000000008264: BF9F0000
	s_code_end                                                 // 000000008268: BF9F0000
	s_code_end                                                 // 00000000826C: BF9F0000
	s_code_end                                                 // 000000008270: BF9F0000
	s_code_end                                                 // 000000008274: BF9F0000
	s_code_end                                                 // 000000008278: BF9F0000
	s_code_end                                                 // 00000000827C: BF9F0000
	s_code_end                                                 // 000000008280: BF9F0000
	s_code_end                                                 // 000000008284: BF9F0000
	s_code_end                                                 // 000000008288: BF9F0000
	s_code_end                                                 // 00000000828C: BF9F0000
	s_code_end                                                 // 000000008290: BF9F0000
	s_code_end                                                 // 000000008294: BF9F0000
	s_code_end                                                 // 000000008298: BF9F0000
	s_code_end                                                 // 00000000829C: BF9F0000
	s_code_end                                                 // 0000000082A0: BF9F0000
	s_code_end                                                 // 0000000082A4: BF9F0000
	s_code_end                                                 // 0000000082A8: BF9F0000
	s_code_end                                                 // 0000000082AC: BF9F0000
	s_code_end                                                 // 0000000082B0: BF9F0000
	s_code_end                                                 // 0000000082B4: BF9F0000
	s_code_end                                                 // 0000000082B8: BF9F0000
	s_code_end                                                 // 0000000082BC: BF9F0000
	s_code_end                                                 // 0000000082C0: BF9F0000
	s_code_end                                                 // 0000000082C4: BF9F0000
	s_code_end                                                 // 0000000082C8: BF9F0000
	s_code_end                                                 // 0000000082CC: BF9F0000
	s_code_end                                                 // 0000000082D0: BF9F0000
	s_code_end                                                 // 0000000082D4: BF9F0000
	s_code_end                                                 // 0000000082D8: BF9F0000
	s_code_end                                                 // 0000000082DC: BF9F0000
	s_code_end                                                 // 0000000082E0: BF9F0000
	s_code_end                                                 // 0000000082E4: BF9F0000
	s_code_end                                                 // 0000000082E8: BF9F0000
	s_code_end                                                 // 0000000082EC: BF9F0000
	s_code_end                                                 // 0000000082F0: BF9F0000
	s_code_end                                                 // 0000000082F4: BF9F0000
	s_code_end                                                 // 0000000082F8: BF9F0000
	s_code_end                                                 // 0000000082FC: BF9F0000
