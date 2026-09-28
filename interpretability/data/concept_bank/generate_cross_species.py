#!/usr/bin/env python3
"""Generate concept_bank_cross_species.tsv from the morphology and transcriptomics banks.

Each concept yields two rows (Homo_sapiens, Mus_musculus) with a species-contextualised
prompt for the text encoder. Run from any directory; paths are relative to this file.
"""

import os
from collections.abc import Callable

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV1_PATH = os.path.join(BASE_DIR, "concept_bank_morphology.tsv")
TSV2_PATH = os.path.join(BASE_DIR, "concept_bank_transcriptomics.tsv")
OUT_PATH = os.path.join(BASE_DIR, "concept_bank_cross_species.tsv")

SPECIES = ("Homo_sapiens", "Mus_musculus")
HUMAN_ONLY_KEYWORDS = ["Gleason", "Orphan Annie", "Paget cells", "DCIS", "STIC"]


def is_human_only(concept_name: str) -> bool:
    """Whether a concept only exists in human pathology (no murine prompt counterpart)."""
    return any(kw in concept_name for kw in HUMAN_ONLY_KEYWORDS) or (
        "serous tubal intraepithelial carcinoma" in concept_name.lower()
    )


# ---------------------------------------------------------------------------
# Marker genes for cell types (concept_name -> (human_markers, mouse_markers))
# Human: ALL CAPS; Mouse: Sentence case (first letter uppercase)
# ---------------------------------------------------------------------------
MARKER_GENES = {
    # ----- T cells -----
    "CD8+ cytotoxic T cell": ("CD8A, CD8B, GZMB, PRF1, IFNG", "Cd8a, Cd8b1, Gzmb, Prf1, Ifng"),
    "CD8+ exhausted T cell": ("CD8A, PDCD1, HAVCR2, LAG3, TOX", "Cd8a, Pdcd1, Havcr2, Lag3, Tox"),
    "CD8+ tissue-resident memory T cell": (
        "CD8A, ITGAE, ZNF683, CXCR6, CD69",
        "Cd8a, Itgae, Zfp683, Cxcr6, Cd69",
    ),
    "CD8+ effector memory T cell": (
        "CD8A, GZMK, EOMES, CCR7, SELL",
        "Cd8a, Gzmk, Eomes, Ccr7, Sell",
    ),
    "CD8+ progenitor exhausted T cell": (
        "CD8A, TCF7, SLAMF6, PDCD1, CXCR5",
        "Cd8a, Tcf7, Slamf6, Pdcd1, Cxcr5",
    ),
    "CD8+ terminally exhausted T cell": (
        "CD8A, ENTPD1, HAVCR2, TOX, LAYN",
        "Cd8a, Entpd1, Havcr2, Tox, Layn",
    ),
    "CD4+ naive T cell": ("CD4, CCR7, SELL, LEF1, TCF7", "Cd4, Ccr7, Sell, Lef1, Tcf7"),
    "CD4+ Th1 cell": ("CD4, TBX21, IFNG, CXCR3, IL12RB2", "Cd4, Tbx21, Ifng, Cxcr3, Il12rb2"),
    "CD4+ Th2 cell": ("CD4, GATA3, IL4, IL5, IL13", "Cd4, Gata3, Il4, Il5, Il13"),
    "CD4+ Th17 cell": ("CD4, RORC, IL17A, CCR6, IL23R", "Cd4, Rorc, Il17a, Ccr6, Il23r"),
    "CD4+ follicular helper T cell": (
        "CD4, CXCR5, BCL6, ICOS, POU2AF1",
        "Cd4, Cxcr5, Bcl6, Icos, Pou2af1",
    ),
    "CD4+ central memory T cell": ("CD4, CCR7, SELL, IL7R, TCF7", "Cd4, Ccr7, Sell, Il7r, Tcf7"),
    "CD4+ cytotoxic T cell": ("CD4, GZMB, PRF1, NKG7, GNLY", "Cd4, Gzmb, Prf1, Nkg7, Gnly"),
    "CD4+ Treg effector": (
        "CD4, FOXP3, IL2RA, CTLA4, TNFRSF18",
        "Cd4, Foxp3, Il2ra, Ctla4, Tnfrsf18",
    ),
    "regulatory T cell": ("FOXP3, IL2RA, CTLA4, IKZF2", "Foxp3, Il2ra, Ctla4, Ikzf2"),
    "gamma-delta T cell": ("TRDV2, TRGV9, TRDC, NKG7, GZMB", "Trdv4, Tcrg-C1, Trdc, Nkg7, Gzmb"),
    "NKT cell": ("CD3E, KLRB1, ZBTB16, SLC4A10, TRAV10", "Cd3e, Klrb1c, Zbtb16, Slc4a10, Trav11"),
    "MAIT cell": (
        "SLC4A10, TRAV1-2, KLRB1, ZBTB16, IL18R1",
        "Slc4a10, Trav1, Klrb1c, Zbtb16, Il18r1",
    ),
    "T cell proliferating state": (
        "MKI67, TOP2A, STMN1, TYMS, CDK1",
        "Mki67, Top2a, Stmn1, Tyms, Cdk1",
    ),
    "T cell interferon-stimulated state": (
        "ISG15, MX1, IFIT1, IFI44L, STAT1",
        "Isg15, Mx1, Ifit1, Ifi44l, Stat1",
    ),
    # ----- B cells -----
    "naive B cell": ("CD19, MS4A1, IGHD, IGHM, TCL1A", "Cd19, Ms4a1, Ighd, Ighm, Tcl1"),
    "memory B cell": ("CD19, MS4A1, CD27, IGHG1, TNFRSF13B", "Cd19, Ms4a1, Cd27, Ighg1, Tnfrsf13b"),
    "germinal center B cell": ("BCL6, AICDA, MME, SUGCT, RGS13", "Bcl6, Aicda, Mme, Sugct, Rgs13"),
    "plasmablast": ("MZB1, XBP1, IRF4, PRDM1, IGHG1", "Mzb1, Xbp1, Irf4, Prdm1, Ighg1"),
    "plasma cell": ("SDC1, MZB1, JCHAIN, XBP1", "Sdc1, Mzb1, Jchain, Xbp1"),
    "regulatory B cell": ("CD19, IL10, CD1D, CD5, IGHM", "Cd19, Il10, Cd1d1, Cd5, Ighm"),
    "marginal zone B cell": ("CD19, CD1C, MZB1, NOTCH2, IGHM", "Cd19, Cd1d1, Mzb1, Notch2, Ighm"),
    "B-1 B cell": ("CD5, IGHM, CD43, SIGLECG, ZNF423", "Cd5, Ighm, Spn, Siglecg, Zfp423"),
    "class-switched memory B cell": (
        "CD27, CD19, IGHG1, IGHA1, GPR183",
        "Cd27, Cd19, Ighg1, Igha, Gpr183",
    ),
    # ----- NK / ILC -----
    "NK cell CD56bright": ("NCAM1, KLRC1, XCL1, SELL, GZMK", "Ncr1, Klrc1, Xcl1, Sell, Gzmk"),
    "NK cell CD56dim": ("NCAM1, FCGR3A, GZMB, PRF1", "Ncr1, Klrb1c, Gzmb, Prf1"),
    "NK CD56dim": ("NCAM1, FCGR3A, GZMB, PRF1", "Ncr1, Klrb1c, Gzmb, Prf1"),
    "adaptive NK cell": ("KLRC2, IFNG, GZMH, FCGR3A, CD2", "Klrc2, Ifng, Gzmb, Fcgr3, Cd2"),
    "tissue-resident NK cell": (
        "EOMES, CD69, ITGAE, CXCR6, NCAM1",
        "Eomes, Cd69, Itgae, Cxcr6, Ncr1",
    ),
    "exhausted NK cell": ("TIGIT, LAG3, PDCD1, HAVCR2, CD96", "Tigit, Lag3, Pdcd1, Havcr2, Cd96"),
    "ILC1": ("TBX21, IFNG, NCR1, IL7R, KLRB1", "Tbx21, Ifng, Ncr1, Il7r, Klrb1c"),
    "ILC2": ("GATA3, IL1RL1, PTGDR2, IL7R, AREG", "Gata3, Il1rl1, Ptgdr2, Il7r, Areg"),
    "ILC3": ("RORC, IL7R, IL22, NCR2, KIT", "Rorc, Il7r, Il22, Ncr2, Kit"),
    "lymphoid tissue inducer cell": ("RORC, IL7R, LTA, LTB, CXCR5", "Rorc, Il7r, Lta, Ltb, Cxcr5"),
    # ----- Monocytes / Macrophages -----
    "classical monocyte": ("CD14, S100A8, S100A9, FCN1, VCAN", "Cd14, S100a8, S100a9, Fcn1, Vcan"),
    "non-classical monocyte": (
        "FCGR3A, CX3CR1, CDKN1C, MS4A7, LILRB2",
        "Fcgr3, Cx3cr1, Cdkn1c, Ms4a7, Lilrb4a",
    ),
    "M1-like pro-inflammatory macrophage": (
        "CD68, NOS2, TNF, IL1B, IL6",
        "Cd68, Nos2, Tnf, Il1b, Il6",
    ),
    "M1-like macrophage": ("CD68, NOS2, TNF, IL1B, IL6", "Cd68, Nos2, Tnf, Il1b, Il6"),
    "M2-like anti-inflammatory macrophage": ("CD68, CD163, MRC1, ARG1", "Cd68, Cd163, Mrc1, Arg1"),
    "M2-like macrophage": ("CD68, CD163, MRC1, ARG1", "Cd68, Cd163, Mrc1, Arg1"),
    "tumor-associated macrophage": ("CD68, TREM2, SPP1, APOE", "Cd68, Trem2, Spp1, Apoe"),
    "lipid-associated macrophage": ("TREM2, LPL, LIPA, FABP5, CD9", "Trem2, Lpl, Lipa, Fabp5, Cd9"),
    "TREM2-positive macrophage": (
        "TREM2, APOE, C1QA, CD163, SPP1",
        "Trem2, Apoe, C1qa, Cd163, Spp1",
    ),
    "SPP1-positive macrophage": ("SPP1, MMP9, VEGFA, CD68, FN1", "Spp1, Mmp9, Vegfa, Cd68, Fn1"),
    "Kupffer cell": ("CLEC4F, TIMD4, MARCO, CD163, CD68", "Clec4f, Timd4, Marco, Cd163, Cd68"),
    "alveolar macrophage": (
        "MARCO, FABP4, PPARG, MCEMP1, CD68",
        "Marco, Fabp4, Pparg, Mcemp1, Cd68",
    ),
    "homeostatic microglia": (
        "P2RY12, CX3CR1, TMEM119, SALL1, CSF1R",
        "P2ry12, Cx3cr1, Tmem119, Sall1, Csf1r",
    ),
    "disease-associated microglia": (
        "TREM2, APOE, LPL, SPP1, TYROBP",
        "Trem2, Apoe, Lpl, Spp1, Tyrobp",
    ),
    "osteoclast": ("ACP5, CTSK, MMP9, CALCR, DCSTAMP", "Acp5, Ctsk, Mmp9, Calcr, Dcstamp"),
    "tissue-resident macrophage": (
        "CD68, C1QA, C1QB, MRC1, LYVE1",
        "Cd68, C1qa, C1qb, Mrc1, Lyve1",
    ),
    "peritoneal macrophage": (
        "CD68, GATA6, TGFB2, RARRES2, ICAM2",
        "Cd68, Gata6, Tgfb2, Rarres2, Icam2",
    ),
    "red pulp macrophage": ("SPI1, VCAM1, SPIC, CD68, HMOX1", "Spi1, Vcam1, Spic, Cd68, Hmox1"),
    "interstitial macrophage": (
        "CD68, MAFB, CX3CR1, LYVE1, CCR2",
        "Cd68, Mafb, Cx3cr1, Lyve1, Ccr2",
    ),
    "erythrophagocytic macrophage": (
        "CD68, HMOX1, SLC40A1, SPI1, CD163",
        "Cd68, Hmox1, Slc40a1, Spi1, Cd163",
    ),
    # ----- Dendritic cells -----
    "cDC1": ("CLEC9A, XCR1, BATF3, IRF8", "Clec9a, Xcr1, Batf3, Irf8"),
    "cDC2": ("CD1C, FCER1A, CLEC10A, IRF4", "Cd1c, Fcer1a, Clec10a, Irf4"),
    "plasmacytoid dendritic cell": ("LILRA4, IRF7, TCF4, CLEC4C", "Siglech, Irf7, Tcf4, Bst2"),
    "plasmacytoid DC": ("LILRA4, IRF7, TCF4, CLEC4C", "Siglech, Irf7, Tcf4, Bst2"),
    "mature LAMP3-positive dendritic cell": (
        "LAMP3, CCR7, CCL19, IDO1, FSCN1",
        "Lamp3, Ccr7, Ccl19, Ido1, Fscn1",
    ),
    "monocyte-derived dendritic cell": (
        "CD1C, CD14, ITGAX, FLT3, ZBTB46",
        "Cd1c, Cd14, Itgax, Flt3, Zbtb46",
    ),
    "Langerhans cell": ("CD207, CD1A, EPCAM, TROP2, LANGERIN", "Cd207, Cd1d1, Epcam, Trop2, Cd207"),
    "inflammatory dendritic cell": ("CD14, ITGAX, TNF, IL1B, FCN1", "Cd14, Itgax, Tnf, Il1b, Fcn1"),
    "tolerogenic dendritic cell": (
        "IDO1, IL10, PDCD1LG2, CD274, AREG",
        "Ido1, Il10, Pdcd1lg2, Cd274, Areg",
    ),
    "migratory dendritic cell": (
        "CCR7, LAMP3, FSCN1, CCL19, CD80",
        "Ccr7, Lamp3, Fscn1, Ccl19, Cd80",
    ),
    "follicular dendritic cell": ("CR2, CXCL13, FDCSP, CLU, PRNP", "Cr2, Cxcl13, Fdcsp, Clu, Prnp"),
    # ----- Granulocytes / Mast cells -----
    "neutrophil": ("FCGR3B, CXCR2, S100A8, S100A9", "Ly6g, Cxcr2, S100a8, S100a9"),
    "N1 anti-tumor neutrophil": (
        "ICAM1, TNF, TRAIL, ROS1, S100A8",
        "Icam1, Tnf, Tnfsf10, Ros1, S100a8",
    ),
    "N2 pro-tumor neutrophil": (
        "ARG1, CCL2, VEGFA, MMP9, S100A8",
        "Arg1, Ccl2, Vegfa, Mmp9, S100a8",
    ),
    "eosinophil": ("CLC, EPX, PRG2, SIGLEC8, CCR3", "Epx, Prg2, Siglecf, Ccr3, Il5ra"),
    "basophil": ("CPA3, GATA2, HDC, MS4A2, IL4", "Cpa3, Gata2, Hdc, Ms4a2, Il4"),
    "connective tissue mast cell": (
        "TPSAB1, CMA1, CPA3, KIT, MS4A2",
        "Tpsab1, Cma1, Cpa3, Kit, Ms4a2",
    ),
    "mucosal mast cell": ("TPSAB1, TPSB2, KIT, GATA2, IL4", "Tpsab1, Tpsb2, Kit, Gata2, Il4"),
    "myeloid-derived suppressor cell": (
        "S100A8, S100A9, ARG1, CD33, ITGAM",
        "S100a8, S100a9, Arg1, Cd33, Itgam",
    ),
    # ----- Fibroblasts / CAF -----
    "normal quiescent fibroblast": ("DCN, LUM, COL1A1, VIM, TCF21", "Dcn, Lum, Col1a1, Vim, Tcf21"),
    "myofibroblast": ("ACTA2, TAGLN, CNN1, COL1A1, MYH11", "Acta2, Tagln, Cnn1, Col1a1, Myh11"),
    "myofibroblastic CAF": ("ACTA2, FAP, POSTN, COL1A1", "Acta2, Fap, Postn, Col1a1"),
    "inflammatory CAF": ("IL6, CXCL1, CXCL12, PDGFRA", "Il6, Cxcl1, Cxcl12, Pdgfra"),
    "antigen-presenting CAF": (
        "CD74, HLA-DRA, HLA-DPA1, PDGFRA, CLU",
        "Cd74, H2-Aa, H2-Ab1, Pdgfra, Clu",
    ),
    "pericyte": ("PDGFRB, RGS5, CSPG4, ACTA2", "Pdgfrb, Rgs5, Cspg4, Acta2"),
    "adventitial fibroblast": ("PI16, DPT, MFAP5, COL1A1, DCN", "Pi16, Dpt, Mfap5, Col1a1, Dcn"),
    "reticular fibroblast": ("CCL19, CCL21, CXCL13, CLU, PDPN", "Ccl19, Ccl21a, Cxcl13, Clu, Pdpn"),
    "perivascular fibroblast": (
        "PDGFRB, COL1A1, NOTCH3, MCAM, RGS5",
        "Pdgfrb, Col1a1, Notch3, Mcam, Rgs5",
    ),
    "matrix-producing fibroblast": (
        "COL1A1, COL3A1, FN1, POSTN, LOX",
        "Col1a1, Col3a1, Fn1, Postn, Lox",
    ),
    "wound-healing fibroblast": (
        "ACTA2, FN1, COL1A1, TGFB1, CTGF",
        "Acta2, Fn1, Col1a1, Tgfb1, Ctgf",
    ),
    "senescent fibroblast": (
        "CDKN1A, CDKN2A, IL6, CXCL8, SERPINE1",
        "Cdkn1a, Cdkn2a, Il6, Cxcl1, Serpine1",
    ),
    # ----- Endothelial -----
    "arterial endothelial cell": (
        "PECAM1, GJA5, HEY1, EFNB2, BMX",
        "Pecam1, Gja5, Hey1, Efnb2, Bmx",
    ),
    "endothelial (arterial)": ("PECAM1, GJA5, HEY1, EFNB2", "Pecam1, Gja5, Hey1, Efnb2"),
    "venous endothelial cell": (
        "PECAM1, ACKR1, PLVAP, VWF, NR2F2",
        "Pecam1, Ackr1, Plvap, Vwf, Nr2f2",
    ),
    "capillary endothelial cell": (
        "PECAM1, RGCC, CA4, GPIHBP1, CD36",
        "Pecam1, Rgcc, Car4, Gpihbp1, Cd36",
    ),
    "lymphatic endothelial cell": (
        "PROX1, LYVE1, PDPN, FLT4, CCL21",
        "Prox1, Lyve1, Pdpn, Flt4, Ccl21a",
    ),
    "angiogenic tip cell": ("ESM1, CXCR4, APLN, DLL4, KDR", "Esm1, Cxcr4, Apln, Dll4, Kdr"),
    "angiogenic stalk cell": (
        "PECAM1, NOTCH1, JAG1, DLL4, HEY1",
        "Pecam1, Notch1, Jag1, Dll4, Hey1",
    ),
    "high endothelial venule cell": (
        "ACKR1, TREM1, GBP5, CCL21, PECAM1",
        "Ackr1, Trem1, Gbp5, Ccl21a, Pecam1",
    ),
    "tumor endothelial cell": (
        "PECAM1, PLVAP, IGFBP7, INSR, HSPG2",
        "Pecam1, Plvap, Igfbp7, Insr, Hspg2",
    ),
    # ----- Mesenchymal / Stromal -----
    "mesenchymal stem cell": ("ENG, NT5E, THY1, VCAM1, PDGFRB", "Eng, Nt5e, Thy1, Vcam1, Pdgfrb"),
    "adipocyte": ("PPARG, ADIPOQ, LEP, FABP4, PLIN1", "Pparg, Adipoq, Lep, Fabp4, Plin1"),
    "osteoblast": ("BGLAP, RUNX2, SP7, COL1A1, ALPL", "Bglap, Runx2, Sp7, Col1a1, Alpl"),
    "chondrocyte": ("SOX9, COL2A1, ACAN, COL9A1, COMP", "Sox9, Col2a1, Acan, Col9a1, Comp"),
    "smooth muscle cell": ("ACTA2, MYH11, CNN1, TAGLN, SMTN", "Acta2, Myh11, Cnn1, Tagln, Smtn"),
    "hepatic stellate cell": ("LRAT, RGS5, PDGFRB, HGF, DES", "Lrat, Rgs5, Pdgfrb, Hgf, Des"),
    "pancreatic stellate cell": (
        "ACTA2, PDGFRB, COL1A1, DES, VIM",
        "Acta2, Pdgfrb, Col1a1, Des, Vim",
    ),
    "mesothelial cell": ("MSLN, KRT19, WT1, UPK3B, CALB2", "Msln, Krt19, Wt1, Upk3b, Calb2"),
    # ----- Neural -----
    "neuron": ("RBFOX3, SYP, MAP2, SNAP25, NRGN", "Rbfox3, Syp, Map2, Snap25, Nrgn"),
    "Schwann cell": ("MPZ, PMP22, MBP, SOX10, S100B", "Mpz, Pmp22, Mbp, Sox10, S100b"),
    "oligodendrocyte": ("MBP, PLP1, MOG, MAG, CLDN11", "Mbp, Plp1, Mog, Mag, Cldn11"),
    "astrocyte": ("GFAP, AQP4, S100B, ALDH1L1, SLC1A2", "Gfap, Aqp4, S100b, Aldh1l1, Slc1a2"),
    "enteric glial cell": (
        "S100B, SOX10, PLP1, GFAP, GPR37L1",
        "S100b, Sox10, Plp1, Gfap, Gpr37l1",
    ),
    "satellite glial cell": (
        "FABP7, SOX10, S100B, GPR37L1, KCNJ10",
        "Fabp7, Sox10, S100b, Gpr37l1, Kcnj10",
    ),
    # ----- Blood / Hematopoietic -----
    "erythrocyte": ("HBA1, HBA2, HBB, SLC4A1, GYPA", "Hba-a1, Hba-a2, Hbb-bs, Slc4a1, Gypa"),
    "megakaryocyte": ("GP1BA, ITGA2B, PF4, PPBP, TUBB1", "Gp1ba, Itga2b, Pf4, Ppbp, Tubb1"),
    "platelet": ("PF4, PPBP, GP9, ITGA2B, TUBB1", "Pf4, Ppbp, Gp9, Itga2b, Tubb1"),
    "hematopoietic stem cell": ("CD34, KIT, THY1, CXCR4", "Kit, Sca1, Thy1, Cxcr4"),
    "common lymphoid progenitor": (
        "IL7R, FLT3, DNTT, VPREB1, CD34",
        "Il7r, Flt3, Dntt, Vpreb1, Kit",
    ),
    "common myeloid progenitor": ("CD34, KIT, FLT3, MPO, CSF2RA", "Kit, Flt3, Mpo, Csf2ra, Cd34"),
    "granulocyte-monocyte progenitor": (
        "CSF2RA, MPO, ELANE, CEBPA, SPI1",
        "Csf2ra, Mpo, Elane, Cebpa, Spi1",
    ),
    "megakaryocyte-erythrocyte progenitor": (
        "GATA1, KLF1, EPOR, ITGA2B, CD34",
        "Gata1, Klf1, Epor, Itga2b, Kit",
    ),
    # ----- Epithelial (normal) -----
    "luminal epithelial cell": (
        "EPCAM, KRT8, KRT18, KRT19, MUC1",
        "Epcam, Krt8, Krt18, Krt19, Muc1",
    ),
    "basal epithelial cell": (
        "KRT5, KRT14, TP63, ITGA6, ITGB4",
        "Krt5, Krt14, Trp63, Itga6, Itgb4",
    ),
    "secretory epithelial cell": (
        "MUC5AC, MUC5B, SCGB1A1, SCGB3A1, WFDC2",
        "Muc5ac, Muc5b, Scgb1a1, Scgb3a1, Wfdc2",
    ),
    "ciliated epithelial cell": (
        "FOXJ1, DNAH5, CFAP43, RSPH1, TPPP3",
        "Foxj1, Dnah5, Cfap43, Rsph1, Tppp3",
    ),
    "goblet cell": ("MUC2, MUC5AC, TFF3, SPDEF, FCGBP", "Muc2, Muc5ac, Tff3, Spdef, Fcgbp"),
    "type I pneumocyte": ("AGER, PDPN, AQP5, HOPX, EMP2", "Ager, Pdpn, Aqp5, Hopx, Emp2"),
    "type II pneumocyte": (
        "SFTPC, SFTPB, SFTPA1, ABCA3, NAPSA",
        "Sftpc, Sftpb, Sftpa1, Abca3, Napsa",
    ),
    "hepatocyte": ("ALB, APOB, CYP3A4, HNF4A", "Alb, Apob, Cyp3a11, Hnf4a"),
    "cholangiocyte": ("KRT19, KRT7, SOX9, EPCAM, CFTR", "Krt19, Krt7, Sox9, Epcam, Cftr"),
    "enterocyte": ("FABP2, VIL1, SLC26A3, ANPEP", "Fabp2, Vil1, Slc26a3, Anpep"),
    "Paneth cell": ("DEFA5, DEFA6, LYZ, REG3A, MMP7", "Defa1, Defa5, Lyz1, Reg3a, Mmp7"),
    "chief cell": ("PGA3, PGA4, PGA5, PGC, MIST1", "Pga5, Pgc, Bhlha15, Gif, Mist1"),
    "parietal cell": ("ATP4A, ATP4B, GIF, CCKBR, KCNQ1", "Atp4a, Atp4b, Gif, Cckbr, Kcnq1"),
    "urothelial cell": ("UPK1A, UPK2, UPK3A, KRT20, KRT5", "Upk1a, Upk2, Upk3a, Krt20, Krt5"),
    "keratinocyte": ("KRT14, KRT5, KRT10, IVL", "Krt14, Krt5, Krt10, Ivl"),
    # ----- Cancer epithelial states -----
    "cycling cancer cell": ("MKI67, TOP2A, CDK1, CCNB1, AURKA", "Mki67, Top2a, Cdk1, Ccnb1, Aurka"),
    "EMT-like cancer cell": ("VIM, FN1, CDH2, SNAI1, ZEB1", "Vim, Fn1, Cdh2, Snai1, Zeb1"),
    "hypoxic cancer cell": ("HIF1A, VEGFA, CA9, SLC2A1, LDHA", "Hif1a, Vegfa, Car9, Slc2a1, Ldha"),
    "stem-like cancer cell": ("SOX2, NANOG, POU5F1, LGR5, CD44", "Sox2, Nanog, Pou5f1, Lgr5, Cd44"),
    "differentiated cancer cell": (
        "KRT20, MUC2, CDX2, VIL1, TFF3",
        "Krt20, Muc2, Cdx2, Vil1, Tff3",
    ),
    "senescent cancer cell": (
        "CDKN1A, CDKN2A, GLB1, SERPINE1, IL6",
        "Cdkn1a, Cdkn2a, Glb1, Serpine1, Il6",
    ),
    "inflammatory cancer cell": ("IL6, CXCL8, TNF, CCL2, NFKB1", "Il6, Cxcl1, Tnf, Ccl2, Nfkb1"),
    "metabolic cancer cell": ("PKM, LDHA, SLC2A1, IDH1, GAPDH", "Pkm, Ldha, Slc2a1, Idh1, Gapdh"),
    "interferon-responsive cancer cell": (
        "ISG15, MX1, IFIT1, STAT1, OAS1",
        "Isg15, Mx1, Ifit1, Stat1, Oas1a",
    ),
    "stress-response cancer cell": (
        "HSPA1A, HSP90AA1, DDIT3, ATF4, XBP1",
        "Hspa1a, Hsp90aa1, Ddit3, Atf4, Xbp1",
    ),
    "MHC-I high cancer cell": ("HLA-A, HLA-B, HLA-C, B2M, TAP1", "H2-K1, H2-D1, H2-L, B2m, Tap1"),
    "MHC-I low cancer cell": ("HLA-A, B2M, TAP1, NLRC5, ERAP1", "H2-K1, B2m, Tap1, Nlrc5, Erap1"),
    "cancer stem cell": ("CD44, ALDH1A1, PROM1, SOX2", "Cd44, Aldh1a1, Prom1, Sox2"),
    "circulating tumor cell": ("EPCAM, KRT19, KRT8, MUC1, VIM", "Epcam, Krt19, Krt8, Muc1, Vim"),
    "disseminated tumor cell": ("EPCAM, KRT19, KRT8, VIM, ZEB1", "Epcam, Krt19, Krt8, Vim, Zeb1"),
    "drug-resistant cancer cell": (
        "ABCB1, ABCG2, ABCC1, GSTP1, NRF2",
        "Abcb1a, Abcg2, Abcc1, Gstp1, Nfe2l2",
    ),
    "anoikis-resistant cancer cell": (
        "BCL2, BIRC5, ITGB1, CAV1, TRKB",
        "Bcl2, Birc5, Itgb1, Cav1, Ntrk2",
    ),
    "dormant cancer cell": (
        "NR2F1, DEC2, p27KIP1, TGFB2, BMP7",
        "Nr2f1, Bhlhe41, Cdkn1b, Tgfb2, Bmp7",
    ),
    "therapy-induced senescent cancer cell": (
        "CDKN1A, CDKN2A, IL6, SERPINE1, GLB1",
        "Cdkn1a, Cdkn2a, Il6, Serpine1, Glb1",
    ),
    "tumor-initiating cell": (
        "CD44, ALDH1A1, LGR5, BMI1, NANOG",
        "Cd44, Aldh1a1, Lgr5, Bmi1, Nanog",
    ),
    "metastatic-competent cancer cell": (
        "TWIST1, SNAI1, MMP2, CXCR4, ITGB1",
        "Twist1, Snai1, Mmp2, Cxcr4, Itgb1",
    ),
    "immune-evasive cancer cell": (
        "CD274, CTLA4, IDO1, LGALS9, B2M",
        "Cd274, Ctla4, Ido1, Lgals9, B2m",
    ),
    # ----- Organ-specific cell types -----
    "podocyte": ("NPHS1, NPHS2, SYNPO, WT1, PODXL", "Nphs1, Nphs2, Synpo, Wt1, Podxl"),
    "melanocyte": ("MITF, TYR, PMEL, DCT, SOX10", "Mitf, Tyr, Pmel, Dct, Sox10"),
    "thyroid follicular cell": ("TG, TPO, NIS, TSHR, PAX8", "Tg, Tpo, Slc5a5, Tshr, Pax8"),
    "alveolar epithelial progenitor": (
        "SFTPC, NKX2-1, SOX9, AXIN2, LGR5",
        "Sftpc, Nkx2-1, Sox9, Axin2, Lgr5",
    ),
    "intestinal stem cell": ("LGR5, OLFM4, ASCL2, SOX9, EPHB2", "Lgr5, Olfm4, Ascl2, Sox9, Ephb2"),
    "pancreatic acinar cell": (
        "PRSS1, CPA1, CELA3A, AMY2A, PNLIP",
        "Prss1, Cpa1, Cela3a, Amy2a5, Pnlip",
    ),
    "pancreatic beta cell": ("INS, MAFA, PDX1, NKX6-1, GCK", "Ins1, Ins2, Mafa, Pdx1, Nkx6-1"),
    "pancreatic alpha cell": ("GCG, ARX, IRX2, GC, MAFB", "Gcg, Arx, Irx2, Gc, Mafb"),
    "Sertoli cell": ("SOX9, AMH, GATA4, WT1, CLDN11", "Sox9, Amh, Gata4, Wt1, Cldn11"),
    "Leydig cell": (
        "STAR, CYP11A1, HSD3B2, CYP17A1, INSL3",
        "Star, Cyp11a1, Hsd3b1, Cyp17a1, Insl3",
    ),
    "trophoblast": ("KRT7, TFAP2C, GATA3, CGA, HLA-G", "Krt7, Tfap2c, Gata3, Cga, Prl3b1"),
    "decidual cell": ("PRL, IGFBP1, WNT4, FOXO1, DKK1", "Prl8a2, Igfbp1, Wnt4, Foxo1, Dkk1"),
    "club cell": (
        "SCGB1A1, SCGB3A2, CYP2F1, BPIFB1, KRT15",
        "Scgb1a1, Scgb3a2, Cyp2f2, Bpifb1, Krt15",
    ),
    "pulmonary neuroendocrine cell": (
        "CHGA, SYP, ASCL1, GRP, CALCA",
        "Chga, Syp, Ascl1, Grp, Calca",
    ),
    "tuft cell": ("DCLK1, TRPM5, POU2F3, GFI1B, SH2D6", "Dclk1, Trpm5, Pou2f3, Gfi1b, Sh2d6"),
}

# ---------------------------------------------------------------------------
# Program genes for gene_programs (concept_name -> (human_genes, mouse_genes))
# Human: ALL CAPS; Mouse: Sentence case
# ---------------------------------------------------------------------------
PROGRAM_GENES = {
    "epithelial-mesenchymal transition program": (
        "VIM, FN1, SNAI1, ZEB1, CDH2",
        "Vim, Fn1, Snai1, Zeb1, Cdh2",
    ),
    "hypoxia response program": (
        "HIF1A, VEGFA, LDHA, PGK1, SLC2A1",
        "Hif1a, Vegfa, Ldha, Pgk1, Slc2a1",
    ),
    "inflammatory response program": ("IL1B, IL6, TNF, CXCL8, CCL2", "Il1b, Il6, Tnf, Cxcl1, Ccl2"),
    "interferon gamma response program": (
        "STAT1, IRF1, GBP1, IDO1, CXCL10",
        "Stat1, Irf1, Gbp1, Ido1, Cxcl10",
    ),
    "p53 pathway program": (
        "TP53, CDKN1A, MDM2, BAX, GADD45A",
        "Trp53, Cdkn1a, Mdm2, Bax, Gadd45a",
    ),
    "oxidative phosphorylation program": (
        "ATP5F1B, NDUFA1, COX7C, UQCRC2, SDHA",
        "Atp5f1b, Ndufa1, Cox7c, Uqcrc2, Sdha",
    ),
    "glycolysis program": ("HK2, PFKP, PKM, LDHA, ENO1", "Hk2, Pfkp, Pkm, Ldha, Eno1"),
    "T cell exhaustion program": (
        "PDCD1, HAVCR2, LAG3, TOX, TIGIT",
        "Pdcd1, Havcr2, Lag3, Tox, Tigit",
    ),
    "Warburg effect aerobic glycolysis program": (
        "HK2, PKM, LDHA, SLC2A1, PDK1",
        "Hk2, Pkm, Ldha, Slc2a1, Pdk1",
    ),
    "angiogenesis program": ("VEGFA, KDR, FLT1, ANGPT2, DLL4", "Vegfa, Kdr, Flt1, Angpt2, Dll4"),
    # Additional MSigDB hallmark programs
    "TNF-alpha signaling via NF-kB program": (
        "NFKBIA, TNFAIP3, CCL2, CXCL2, JUNB",
        "Nfkbia, Tnfaip3, Ccl2, Cxcl2, Junb",
    ),
    "interferon alpha response program": (
        "MX1, ISG15, IFIT1, OAS1, STAT2",
        "Mx1, Isg15, Ifit1, Oas1a, Stat2",
    ),
    "MYC targets V1 program": ("NPM1, NME1, LDHA, PHB, SRM", "Npm1, Nme1, Ldha, Phb, Srm"),
    "MYC targets V2 program": (
        "NOP56, MRTO4, SLC19A1, POLR3G, NOP16",
        "Nop56, Mrto4, Slc19a1, Polr3g, Nop16",
    ),
    "E2F targets program": ("PCNA, MCM6, RRM2, CDK1, POLA1", "Pcna, Mcm6, Rrm2, Cdk1, Pola1"),
    "G2M checkpoint program": (
        "CDK1, CCNB2, TOP2A, AURKA, BUB1",
        "Cdk1, Ccnb2, Top2a, Aurka, Bub1",
    ),
    "apoptosis program": ("BCL2, BAX, CASP3, CASP9, CYCS", "Bcl2, Bax, Casp3, Casp9, Cycs"),
    "complement program": ("C1QA, C1QB, C3, C4A, CFB", "C1qa, C1qb, C3, C4b, Cfb"),
    "coagulation program": ("F2, FGA, FGB, SERPINC1, PROC", "F2, Fga, Fgb, Serpinc1, Proc"),
    "TGF-beta signaling program": (
        "TGFB1, SMAD3, SMAD4, CDKN2B, SERPINE1",
        "Tgfb1, Smad3, Smad4, Cdkn2b, Serpine1",
    ),
    "Wnt-beta-catenin signaling program": (
        "CTNNB1, MYC, CCND1, TCF7L2, AXIN2",
        "Ctnnb1, Myc, Ccnd1, Tcf7l2, Axin2",
    ),
    "Notch signaling program": ("NOTCH1, HES1, HEY1, JAG1, DLL1", "Notch1, Hes1, Hey1, Jag1, Dll1"),
    "Hedgehog signaling program": ("SHH, PTCH1, SMO, GLI1, GLI2", "Shh, Ptch1, Smo, Gli1, Gli2"),
    "KRAS signaling up program": (
        "DUSP6, SPRY2, ETV4, ETV5, PHLDA1",
        "Dusp6, Spry2, Etv4, Etv5, Phlda1",
    ),
    "mTORC1 signaling program": (
        "RPS6KB1, EIF4EBP1, SLC7A5, SCD, ACLY",
        "Rps6kb1, Eif4ebp1, Slc7a5, Scd1, Acly",
    ),
    "PI3K-AKT-mTOR signaling program": (
        "AKT1, PIK3CA, MTOR, PTEN, RPS6",
        "Akt1, Pik3ca, Mtor, Pten, Rps6",
    ),
    "unfolded protein response program": (
        "XBP1, ATF4, DDIT3, HSPA5, ERN1",
        "Xbp1, Atf4, Ddit3, Hspa5, Ern1",
    ),
    "reactive oxygen species pathway program": (
        "SOD1, SOD2, CAT, GPX1, TXN",
        "Sod1, Sod2, Cat, Gpx1, Txn1",
    ),
    "fatty acid metabolism program": (
        "FASN, ACACA, SCD, FADS2, HADHA",
        "Fasn, Acaca, Scd1, Fads2, Hadha",
    ),
    "cholesterol homeostasis program": (
        "HMGCR, LDLR, SREBF2, SQLE, DHCR7",
        "Hmgcr, Ldlr, Srebf2, Sqle, Dhcr7",
    ),
    "IL2-STAT5 signaling program": (
        "IL2RA, FOXP3, BCL2, MYC, CISH",
        "Il2ra, Foxp3, Bcl2, Myc, Cish",
    ),
    "IL6-JAK-STAT3 signaling program": (
        "STAT3, SOCS3, MYC, BCL2L1, PIM1",
        "Stat3, Socs3, Myc, Bcl2l1, Pim1",
    ),
    "DNA repair program": (
        "BRCA1, BRCA2, RAD51, XRCC1, PARP1",
        "Brca1, Brca2, Rad51, Xrcc1, Parp1",
    ),
    "mitotic spindle program": (
        "AURKA, AURKB, KIF11, TPX2, BUB1B",
        "Aurka, Aurkb, Kif11, Tpx2, Bub1b",
    ),
    "adipogenesis program": (
        "PPARG, CEBPA, FABP4, ADIPOQ, PLIN1",
        "Pparg, Cebpa, Fabp4, Adipoq, Plin1",
    ),
    "myogenesis program": ("MYOD1, MYOG, MYF5, MYH1, DES", "Myod1, Myog, Myf5, Myh1, Des"),
    # Immune programs
    "T cell activation program": (
        "CD69, IL2, TNFRSF9, IFNG, CD25",
        "Cd69, Il2, Tnfrsf9, Ifng, Il2ra",
    ),
    "T cell cytotoxicity program": (
        "GZMB, PRF1, GNLY, NKG7, FASLG",
        "Gzmb, Prf1, Gnly, Nkg7, Fasl",
    ),
    "Treg suppressive program": (
        "FOXP3, CTLA4, IL10, TGFB1, IL35",
        "Foxp3, Ctla4, Il10, Tgfb1, Ebi3",
    ),
    "Th1 polarization program": (
        "TBX21, IFNG, IL12RB2, CXCR3, STAT4",
        "Tbx21, Ifng, Il12rb2, Cxcr3, Stat4",
    ),
    "Th2 polarization program": ("GATA3, IL4, IL5, IL13, STAT6", "Gata3, Il4, Il5, Il13, Stat6"),
    "Th17 polarization program": (
        "RORC, IL17A, IL22, CCR6, IL23R",
        "Rorc, Il17a, Il22, Ccr6, Il23r",
    ),
    "B cell activation program": ("CD19, MS4A1, CD86, AICDA, MYC", "Cd19, Ms4a1, Cd86, Aicda, Myc"),
    "germinal center reaction program": (
        "BCL6, AICDA, MYC, FOXO1, CXCR4",
        "Bcl6, Aicda, Myc, Foxo1, Cxcr4",
    ),
    "plasma cell differentiation program": (
        "PRDM1, IRF4, XBP1, SDC1, MZB1",
        "Prdm1, Irf4, Xbp1, Sdc1, Mzb1",
    ),
    "antibody secretion program": (
        "JCHAIN, XBP1, MZB1, IGHG1, IGHA1",
        "Jchain, Xbp1, Mzb1, Ighg1, Igha",
    ),
    "M1 macrophage activation program": (
        "NOS2, TNF, IL1B, IL6, CXCL10",
        "Nos2, Tnf, Il1b, Il6, Cxcl10",
    ),
    "M2 macrophage activation program": (
        "MRC1, ARG1, CD163, IL10, TGFB1",
        "Mrc1, Arg1, Cd163, Il10, Tgfb1",
    ),
    "phagocytosis program": ("MERTK, AXL, TYRO3, CD47, SIRPA", "Mertk, Axl, Tyro3, Cd47, Sirpa"),
    "antigen processing program": (
        "TAP1, TAP2, PSMB8, PSMB9, B2M",
        "Tap1, Tap2, Psmb8, Psmb9, B2m",
    ),
    "cross-presentation program": (
        "TAP1, SEC61A1, WDFY4, RAB11A, CLEC9A",
        "Tap1, Sec61a1, Wdfy4, Rab11a, Clec9a",
    ),
    "NK cell activation program": (
        "KLRK1, NCR1, GZMB, PRF1, IFNG",
        "Klrk1, Ncr1, Gzmb, Prf1, Ifng",
    ),
    "NK cell exhaustion program": (
        "TIGIT, LAG3, PDCD1, HAVCR2, CD96",
        "Tigit, Lag3, Pdcd1, Havcr2, Cd96",
    ),
    "neutrophil degranulation program": (
        "ELANE, MPO, LTF, MMP9, CTSG",
        "Elane, Mpo, Ltf, Mmp9, Ctsg",
    ),
    "neutrophil extracellular trap formation program": (
        "MPO, ELANE, PADI4, H3F3A, HMGB1",
        "Mpo, Elane, Padi4, H3f3a, Hmgb1",
    ),
    "complement activation program": ("C1QA, C3, C5, CFB, MASP2", "C1qa, C3, C5, Cfb, Masp2"),
    "cytokine storm program": ("IL6, IL1B, TNF, IFNG, CXCL10", "Il6, Il1b, Tnf, Ifng, Cxcl10"),
    "chemokine signaling program": (
        "CCL2, CXCL12, CCR7, CXCR4, CCL19",
        "Ccl2, Cxcl12, Ccr7, Cxcr4, Ccl19",
    ),
    "PD-1 PD-L1 checkpoint program": (
        "PDCD1, CD274, PDCD1LG2, LAG3, HAVCR2",
        "Pdcd1, Cd274, Pdcd1lg2, Lag3, Havcr2",
    ),
    "CTLA-4 checkpoint program": ("CTLA4, CD80, CD86, CD28, ICOS", "Ctla4, Cd80, Cd86, Cd28, Icos"),
    "TIM-3 LAG-3 checkpoint program": (
        "HAVCR2, LAG3, LGALS9, FGL1, CEACAM1",
        "Havcr2, Lag3, Lgals9, Fgl1, Ceacam1",
    ),
    "type I interferon response program": (
        "IFNB1, MX1, OAS1, ISG15, IRF3",
        "Ifnb1, Mx1, Oas1a, Isg15, Irf3",
    ),
    "type II interferon response program": (
        "IFNG, STAT1, IRF1, GBP1, CXCL10",
        "Ifng, Stat1, Irf1, Gbp1, Cxcl10",
    ),
    "toll-like receptor signaling program": (
        "TLR4, MYD88, TRAF6, IRAK4, NF-kB",
        "Tlr4, Myd88, Traf6, Irak4, Nfkb1",
    ),
    "inflammasome activation program": (
        "NLRP3, CASP1, IL1B, IL18, PYCARD",
        "Nlrp3, Casp1, Il1b, Il18, Pycard",
    ),
    "T cell trafficking program": (
        "CCR7, CXCR3, CXCL9, CXCL10, ICAM1",
        "Ccr7, Cxcr3, Cxcl9, Cxcl10, Icam1",
    ),
    "immune evasion program": (
        "CD274, IDO1, LGALS9, VTCN1, NT5E",
        "Cd274, Ido1, Lgals9, Vtcn1, Nt5e",
    ),
    "myeloid recruitment program": (
        "CCL2, CSF1, CXCL12, IL34, CCL5",
        "Ccl2, Csf1, Cxcl12, Il34, Ccl5",
    ),
    "tertiary lymphoid structure formation program": (
        "CXCL13, CCL19, CCL21, LTA, LTB",
        "Cxcl13, Ccl19, Ccl21a, Lta, Ltb",
    ),
    # Metabolic programs
    "oxidative phosphorylation metabolic program": (
        "NDUFA1, NDUFB1, COX7C, ATP5F1B, UQCRC2",
        "Ndufa1, Ndufb1, Cox7c, Atp5f1b, Uqcrc2",
    ),
    "fatty acid synthesis program": (
        "FASN, ACACA, SCD, ACLY, ELOVL6",
        "Fasn, Acaca, Scd1, Acly, Elovl6",
    ),
    "beta-oxidation program": (
        "CPT1A, ACADM, HADHA, HADHB, ACOX1",
        "Cpt1a, Acadm, Hadha, Hadhb, Acox1",
    ),
    "glutamine metabolism program": (
        "GLS, GLUL, SLC1A5, GOT2, ASNS",
        "Gls, Glul, Slc1a5, Got2, Asns",
    ),
    "one-carbon metabolism program": (
        "MTHFD1, SHMT2, DHFR, TYMS, MTHFR",
        "Mthfd1, Shmt2, Dhfr, Tyms, Mthfr",
    ),
    "nucleotide biosynthesis program": (
        "CAD, DHODH, UMPS, CTPS1, IMPDH2",
        "Cad, Dhodh, Umps, Ctps, Impdh2",
    ),
    "amino acid catabolism program": (
        "GOT1, GOT2, GPT, BCAT1, IDO1",
        "Got1, Got2, Gpt, Bcat1, Ido1",
    ),
    "tryptophan-kynurenine metabolism program": (
        "IDO1, TDO2, KMO, KYNU, HAAO",
        "Ido1, Tdo2, Kmo, Kynu, Haao",
    ),
    "arginine metabolism program": ("ARG1, ARG2, NOS2, ASS1, ASL", "Arg1, Arg2, Nos2, Ass1, Asl"),
    "lipid peroxidation ferroptosis program": (
        "GPX4, SLC7A11, ACSL4, LPCAT3, ALOX15",
        "Gpx4, Slc7a11, Acsl4, Lpcat3, Alox15",
    ),
    "mitochondrial biogenesis program": (
        "PPARGC1A, TFAM, NRF1, POLG, TOMM20",
        "Ppargc1a, Tfam, Nrf1, Polg, Tomm20",
    ),
    "autophagy program": (
        "BECN1, ATG5, ATG7, MAP1LC3B, SQSTM1",
        "Becn1, Atg5, Atg7, Map1lc3b, Sqstm1",
    ),
    "pentose phosphate pathway program": (
        "G6PD, PGD, TKT, TALDO1, RPIA",
        "G6pdx, Pgd, Tkt, Taldo1, Rpia",
    ),
    "TCA cycle program": ("CS, ACO2, IDH2, OGDH, SDHA", "Cs, Aco2, Idh2, Ogdh, Sdha"),
    # Remodeling programs
    "ECM production program": (
        "COL1A1, COL3A1, FN1, POSTN, SPARC",
        "Col1a1, Col3a1, Fn1, Postn, Sparc",
    ),
    "ECM degradation MMP program": (
        "MMP2, MMP9, MMP14, TIMP1, TIMP2",
        "Mmp2, Mmp9, Mmp14, Timp1, Timp2",
    ),
    "collagen deposition program": (
        "COL1A1, COL1A2, COL3A1, COL5A1, LOX",
        "Col1a1, Col1a2, Col3a1, Col5a1, Lox",
    ),
    "fibronectin assembly program": (
        "FN1, ITGA5, ITGB1, TNC, THBS1",
        "Fn1, Itga5, Itgb1, Tnc, Thbs1",
    ),
    "TGF-beta response program": (
        "TGFB1, SMAD2, SMAD3, CTGF, SERPINE1",
        "Tgfb1, Smad2, Smad3, Ctgf, Serpine1",
    ),
    "wound healing program": ("FN1, COL1A1, TGFB1, ACTA2, MMP2", "Fn1, Col1a1, Tgfb1, Acta2, Mmp2"),
    "fibrosis program": ("COL1A1, ACTA2, TGFB1, CTGF, LOX", "Col1a1, Acta2, Tgfb1, Ctgf, Lox"),
    "angiogenic program": ("VEGFA, KDR, FLT1, ANGPT2, DLL4", "Vegfa, Kdr, Flt1, Angpt2, Dll4"),
    "integrin signaling program": (
        "ITGB1, ITGAV, ITGA5, ILK, PXN",
        "Itgb1, Itgav, Itga5, Ilk, Pxn",
    ),
    "focal adhesion program": ("PTK2, PXN, VCL, TLN1, ITGB1", "Ptk2, Pxn, Vcl, Tln1, Itgb1"),
    # Cancer hallmark programs
    "sustained proliferative signaling program": (
        "EGFR, ERBB2, MYC, CCND1, CDK4",
        "Egfr, Erbb2, Myc, Ccnd1, Cdk4",
    ),
    "invasion and motility program": (
        "MMP2, MMP9, TWIST1, SNAI1, VIM",
        "Mmp2, Mmp9, Twist1, Snai1, Vim",
    ),
    "metastasis colonization program": (
        "CXCR4, TWIST1, S100A4, MMP9, SNAI2",
        "Cxcr4, Twist1, S100a4, Mmp9, Snai2",
    ),
    "immune evasion cancer hallmark program": (
        "CD274, CTLA4, IDO1, LGALS9, VTCN1",
        "Cd274, Ctla4, Ido1, Lgals9, Vtcn1",
    ),
    "genome instability program": (
        "TP53, BRCA1, ATM, CHEK2, MLH1",
        "Trp53, Brca1, Atm, Chek2, Mlh1",
    ),
    "tumor-promoting inflammation program": (
        "IL6, TNF, CCL2, CXCL8, STAT3",
        "Il6, Tnf, Ccl2, Cxcl1, Stat3",
    ),
    "apoptosis resistance program": (
        "BCL2, BCL2L1, MCL1, BIRC5, XIAP",
        "Bcl2, Bcl2l1, Mcl1, Birc5, Xiap",
    ),
    "drug efflux ABC transporter program": (
        "ABCB1, ABCG2, ABCC1, ABCC2, ABCB4",
        "Abcb1a, Abcg2, Abcc1, Abcc2, Abcb4",
    ),
    "homologous recombination repair program": (
        "BRCA1, BRCA2, RAD51, PALB2, BARD1",
        "Brca1, Brca2, Rad51, Palb2, Bard1",
    ),
    "mismatch repair deficiency program": (
        "MLH1, MSH2, MSH6, PMS2, EPCAM",
        "Mlh1, Msh2, Msh6, Pms2, Epcam",
    ),
    # Cancer meta-programs (Gavish 2023)
    "cell cycle G1/S meta-program": (
        "PCNA, MCM6, RRM2, CDT1, CDC6",
        "Pcna, Mcm6, Rrm2, Cdt1, Cdc6",
    ),
    "cell cycle G2/M meta-program": (
        "CDK1, CCNB1, TOP2A, AURKA, BUB1",
        "Cdk1, Ccnb1, Top2a, Aurka, Bub1",
    ),
    "EMT-I partial EMT meta-program": (
        "LAMC2, LAMB3, ITGA6, KRT17, S100A2",
        "Lamc2, Lamb3, Itga6, Krt17, S100a2",
    ),
    "EMT-II full mesenchymal meta-program": (
        "VIM, FN1, CDH2, SNAI2, COL1A1",
        "Vim, Fn1, Cdh2, Snai2, Col1a1",
    ),
    "hypoxia meta-program": ("HIF1A, VEGFA, CA9, SLC2A1, ADM", "Hif1a, Vegfa, Car9, Slc2a1, Adm"),
    "stress and p53 meta-program": (
        "TP53, CDKN1A, GADD45A, BTG2, MDM2",
        "Trp53, Cdkn1a, Gadd45a, Btg2, Mdm2",
    ),
    "MYC targets meta-program": ("NPM1, NME1, NCL, LDHA, ENO1", "Npm1, Nme1, Ncl, Ldha, Eno1"),
    "interferon response meta-program": (
        "ISG15, MX1, IFIT1, IFI6, OAS1",
        "Isg15, Mx1, Ifit1, Ifi6, Oas1a",
    ),
    "stem-like meta-program": (
        "SOX2, POU5F1, NANOG, CD44, ALDH1A1",
        "Sox2, Pou5f1, Nanog, Cd44, Aldh1a1",
    ),
    "immune evasion meta-program": (
        "CD274, B2M, HLA-A, LGALS9, IDO1",
        "Cd274, B2m, H2-K1, Lgals9, Ido1",
    ),
}

# ---------------------------------------------------------------------------
# Prompt template functions
# ---------------------------------------------------------------------------


def _species_word(species: str) -> str:
    """Return 'human' or 'murine'."""
    return "human" if species == "Homo_sapiens" else "murine"


MORPHOLOGY_TEMPLATES = {
    "normal_tissue": "Normal histology: {name} in H&E stained {sw} tissue",
    "organ_specific": "Histopathological feature: {name} in {sw} {organ} tissue on H&E stain",
    "microenvironment": "Tissue microenvironment: {name} in {sw} tissue on H&E stain",
    "artifact": "Histological artifact: {name} in H&E stained {sw} tissue",
}
MORPHOLOGY_DEFAULT = "Histopathological feature: {name} observed in H&E stained {sw} tissue"


def morphology_prompt(row: pd.Series, species: str) -> str:
    """Prompt for an H1 morphology concept, phrased by its category."""
    template = MORPHOLOGY_TEMPLATES.get(row["category"], MORPHOLOGY_DEFAULT)
    return template.format(name=row["concept_name"], sw=_species_word(species), organ=row["organ"])


def cell_type_prompt(row: pd.Series, species: str) -> str:
    """Prompt for a cell type, with species-specific marker genes when known."""
    sw, name = _species_word(species), row["concept_name"]
    markers = MARKER_GENES.get(name)
    if markers:
        marker_list = markers[SPECIES.index(species)]
        return f"Cell type present in {sw} tissue: {name}, expressing {marker_list}"
    return f"Cell type present in {sw} tissue: {name}"


def niche_prompt(row: pd.Series, species: str) -> str:
    """Prompt for a spatial tissue niche."""
    return f"Spatial tissue niche in {_species_word(species)} tissue: {row['concept_name']}"


def gene_program_prompt(row: pd.Series, species: str) -> str:
    """Prompt for a transcriptional program, with representative genes when known."""
    sw, name = _species_word(species), row["concept_name"]
    genes = PROGRAM_GENES.get(name)
    if genes:
        return (
            f"Transcriptional program active in {sw} cells: {name}, "
            f"involving genes such as {genes[SPECIES.index(species)]}"
        )
    return f"Transcriptional program active in {sw} cells: {name}"


def pathway_prompt(row: pd.Series, species: str) -> str:
    """Prompt for a signalling pathway."""
    return f"Signaling pathway in {_species_word(species)} cells: {row['concept_name']}"


TRANSCRIPTOMIC_TIERS: dict[str, tuple[str, Callable[[pd.Series, str], str]]] = {
    "cell_type": ("H2_cell_type", cell_type_prompt),
    "niche": ("H2_niche", niche_prompt),
    "gene_program": ("H2_gene_program", gene_program_prompt),
    "pathway": ("H2_pathway", pathway_prompt),
}


def _rows(concept: pd.Series, hierarchy: str, prompt: Callable[[str], str]) -> list[dict[str, str]]:
    """One output row per species for a concept."""
    status = "human_only" if is_human_only(concept["concept_name"]) else "shared"
    return [
        {
            "concept_id": concept["concept_id"],
            "concept_name": concept["concept_name"],
            "hierarchy": hierarchy,
            "species": sp,
            "prompt_text": prompt(sp),
            "species_status": status,
        }
        for sp in SPECIES
    ]


def main() -> None:
    """Build the cross-species prompt table and print coverage statistics."""
    morph = pd.read_csv(TSV1_PATH, sep="\t")
    trans = pd.read_csv(TSV2_PATH, sep="\t")

    rows: list[dict[str, str]] = []
    for _, r in morph.iterrows():
        rows += _rows(r, "H1_morphology", lambda sp, r=r: morphology_prompt(r, sp))
    for _, r in trans.iterrows():
        ctype = r["concept_type"]
        if ctype in TRANSCRIPTOMIC_TIERS:
            hierarchy, prompt_fn = TRANSCRIPTOMIC_TIERS[ctype]
            rows += _rows(r, hierarchy, lambda sp, r=r, fn=prompt_fn: fn(r, sp))
        else:
            rows += _rows(
                r,
                f"H2_{ctype}",
                lambda sp, r=r: f"{r['concept_name']} in {_species_word(sp)} tissue",
            )

    columns = [
        "concept_id",
        "concept_name",
        "hierarchy",
        "species",
        "prompt_text",
        "species_status",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT_PATH, sep="\t", index=False)

    print(f"Output written to: {OUT_PATH}")
    print(f"Total rows: {len(df)}")
    for col in ("species", "hierarchy", "species_status"):
        print(f"\nBy {col}:")
        print(df[col].value_counts().to_string())

    n_morph, n_trans = len(morph), len(trans)
    n_total = n_morph + n_trans
    print(f"\nUnique concepts: {n_total} ({n_morph} morphology + {n_trans} transcriptomics)")
    print(f"Expected rows: {n_total * 2} (2 species per concept)")

    cell_types = trans.loc[trans["concept_type"] == "cell_type", "concept_name"]
    programs = trans.loc[trans["concept_type"] == "gene_program", "concept_name"]
    n_markers = int(cell_types.isin(MARKER_GENES.keys()).sum())
    n_program_genes = int(programs.isin(PROGRAM_GENES.keys()).sum())
    print(
        f"\nCell type marker coverage: {n_markers}/{len(cell_types)} cell types have marker genes"
    )
    print(
        f"Gene program gene coverage: {n_program_genes}/{len(programs)} "
        "gene programs have representative genes"
    )

    human_only = df.loc[df["species_status"] == "human_only", "concept_name"].unique()
    print(f"\nHuman-only concepts: {len(human_only)}")
    for name in human_only:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
