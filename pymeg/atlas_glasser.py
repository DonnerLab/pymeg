import mne
import os


'''
Get HCP MMP Parcellation for a set of subjects
'''


def get_hcp(subjects_dir):
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=subjects_dir, verbose=True)


def get_hcp_annotation(subjects_dir, subject):
    for hemi in ['lh', 'rh']:
        # transform atlas to individual space:
        cmd = 'mris_apply_reg --src-annot {} --trg {} --streg {} {}'.format(
            os.path.join(subjects_dir, 'fsaverage', 'label',
                         '{}.HCPMMP1.annot'.format(hemi)),
            os.path.join(subjects_dir, subject, 'label',
                         '{}.HCPMMP1.annot'.format(hemi)),
            os.path.join(subjects_dir, 'fsaverage', 'surf',
                         '{}.sphere.reg'.format(hemi)),
            os.path.join(subjects_dir, subject, 'surf', '{}.sphere.reg'.format(hemi)),)
        os.system(cmd)

def get_hcp_labels(subjects_dir, subjects):
    '''
    Downloads HCP MMP Parcellation and applies it to a set of subjects

    Arguments
    =========
    subjects_dir: str
        Path of freesurfer subjects dir
    subjects: list
        List of subject IDs (need to correspond to folders in 
        freesurfer subject dir.)
    '''

    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=subjects_dir, verbose=True)

    for subj in subjects:
        for hemi in ['lh', 'rh']:

            # transform atlas to individual space:
            cmd = 'mris_apply_reg --src-annot {} --trg {} --streg {} {}'.format(
                os.path.join(subjects_dir, 'fsaverage', 'label',
                             '{}.HCPMMP1_combined.annot'.format(hemi)),
                os.path.join(subjects_dir, subj, 'label',
                             '{}.HCPMMP1_combined.annot'.format(hemi)),
                os.path.join(subjects_dir, 'fsaverage', 'surf',
                             '{}.sphere.reg'.format(hemi)),
                os.path.join(subjects_dir, subj, 'surf', '{}.sphere.reg'.format(hemi)),)
            os.system(cmd)

            # unpack into labels:
            cmd = 'mri_annotation2label --subject {} --hemi {} --labelbase {} --annotation {}'.format(
                subj,
                hemi,
                '{}.HCPMMP1_combined'.format(hemi),
                'HCPMMP1_combined'.format(hemi),
            )
            os.system(cmd)

            # rename in alphabetical order...
            orig_names = [
                '???',
                'Anterior Cingulate and Medial Prefrontal Cortex',
                'Auditory Association Cortex',
                'Dorsal Stream Visual Cortex',
                'DorsoLateral Prefrontal Cortex',
                'Early Auditory Cortex',
                'Early Visual Cortex',
                'Inferior Frontal Cortex',
                'Inferior Parietal Cortex',
                'Insular and Frontal Opercular Cortex',
                'Lateral Temporal Cortex',
                'MT+ Complex and Neighboring Visual Areas',
                'Medial Temporal Cortex',
                'Orbital and Polar Frontal Cortex',
                'Paracentral Lobular and Mid Cingulate Cortex',
                'Posterior Cingulate Cortex',
                'Posterior Opercular Cortex',
                'Premotor Cortex',
                'Primary Visual Cortex (V1)',
                'Somatosensory and Motor Cortex',
                'Superior Parietal Cortex',
                'Temporo-Parieto-Occipital Junction',
                'Ventral Stream Visual Cortex'
            ]

            new_names = [
                '23_inside',
                '19_cingulate_anterior_prefrontal_medial',
                '11_auditory_association',
                '03_visual_dors',
                '22_prefrontal_dorsolateral',
                '10_auditory_primary',
                '02_visual_early',
                '21_frontal_inferior',
                '17_parietal_inferior',
                '12_insular_frontal_opercular',
                '14_lateral_temporal',
                '05_visual_lateral',
                '13_temporal_medial',
                '20_frontal_orbital_polar',
                '07_paracentral_lob_mid_cingulate',
                '18_cingulate_posterior',
                '09_opercular_posterior',
                '08_premotor',
                '01_visual_primary',
                '06_somatosensory_motor',
                '16_parietal_superior',
                '15_temporal_parietal_occipital_junction',
                '04_visual_ventral',
            ]

            for o, n, i in zip(orig_names, new_names,
                               ["%.2d" % i for i in range(23)]):
                os.rename(
                    os.path.join(subjects_dir, subj, 'label',
                                 '{}.HCPMMP1_combined-0{}.label'.format(hemi, i)),
                    os.path.join(subjects_dir, subj, 'label',
                                 '{}.HCPMMP1_{}.label'.format(hemi, o)),
                )
                os.rename(
                    os.path.join(subjects_dir, subj, 'label',
                                 '{}.HCPMMP1_{}.label'.format(hemi, o)),
                    os.path.join(subjects_dir, subj, 'label',
                                 '{}.HCPMMP1_{}.label'.format(hemi, n)),
                )

def get_clusters():

    visual_field_clusters = {
         'vfcvisual':   (
                        u'lh.wang2015atlas.V1d-lh', u'rh.wang2015atlas.V1d-rh',
                        u'lh.wang2015atlas.V1v-lh', u'rh.wang2015atlas.V1v-rh',
                        u'lh.wang2015atlas.V2d-lh', u'rh.wang2015atlas.V2d-rh',
                        u'lh.wang2015atlas.V2v-lh', u'rh.wang2015atlas.V2v-rh',
                        u'lh.wang2015atlas.V3d-lh', u'rh.wang2015atlas.V3d-rh',
                        u'lh.wang2015atlas.V3v-lh', u'rh.wang2015atlas.V3v-rh',
                        u'lh.wang2015atlas.hV4-lh', u'rh.wang2015atlas.hV4-rh',
                        ),
         'vfcVO':       (
                        u'lh.wang2015atlas.VO1-lh', u'rh.wang2015atlas.VO1-rh', 
                        u'lh.wang2015atlas.VO2-lh', u'rh.wang2015atlas.VO2-rh',
                        ),
         'vfcPHC':      (
                        u'lh.wang2015atlas.PHC1-lh', u'rh.wang2015atlas.PHC1-rh',
                        u'lh.wang2015atlas.PHC2-lh', u'rh.wang2015atlas.PHC2-rh',
                        ),
         'vfcV3ab':     (
                        u'lh.wang2015atlas.V3A-lh', u'rh.wang2015atlas.V3A-rh', 
                        u'lh.wang2015atlas.V3B-lh', u'rh.wang2015atlas.V3B-rh',
                        ),
         'vfcTO':       (
                        u'lh.wang2015atlas.TO1-lh', u'rh.wang2015atlas.TO1-rh', 
                        u'lh.wang2015atlas.TO2-lh', u'rh.wang2015atlas.TO2-rh',
                        ),
         'vfcLO':       (
                        u'lh.wang2015atlas.LO1-lh', u'rh.wang2015atlas.LO1-rh', 
                        u'lh.wang2015atlas.LO2-lh', u'rh.wang2015atlas.LO2-rh',
                        ),
         'vfcIPS01':    (
                        u'lh.wang2015atlas.IPS0-lh', u'rh.wang2015atlas.IPS0-rh', 
                        u'lh.wang2015atlas.IPS1-lh', u'rh.wang2015atlas.IPS1-rh',
                        ),
         'vfcIPS2345':  (
                        u'lh.wang2015atlas.IPS2-lh', u'rh.wang2015atlas.IPS2-rh', 
                        u'lh.wang2015atlas.IPS3-lh', u'rh.wang2015atlas.IPS3-rh',
                        u'lh.wang2015atlas.IPS4-lh', u'rh.wang2015atlas.IPS4-rh', 
                        u'lh.wang2015atlas.IPS5-lh', u'rh.wang2015atlas.IPS5-rh',
                        ),
         'vfcSPL':      (
                        u'lh.wang2015atlas.SPL1-lh', u'rh.wang2015atlas.SPL1-rh',
                        ),
         'vfcFEF':      (
                        u'lh.wang2015atlas.FEF-lh', u'rh.wang2015atlas.FEF-rh',
                        ),
         }

    glasser_clusters = {
         'HCPMMP1_visual_primary':                  ('lh.HCPMMP1_01_visual_primary-lh', 'rh.HCPMMP1_01_visual_primary-rh',),
         'HCPMMP1_visual_early':                    ('lh.HCPMMP1_02_visual_early-lh', 'rh.HCPMMP1_02_visual_early-rh',),
         'HCPMMP1_visual_dors':                     ('lh.HCPMMP1_03_visual_dors-lh', 'rh.HCPMMP1_03_visual_dors-rh',),
         'HCPMMP1_visual_ventral':                  ('lh.HCPMMP1_04_visual_ventral-lh', 'rh.HCPMMP1_04_visual_ventral-rh',),
         'HCPMMP1_visual_lateral':                  ('lh.HCPMMP1_05_visual_lateral-lh', 'rh.HCPMMP1_05_visual_lateral-rh',),
         'HCPMMP1_somatosensory_motor':             ('lh.HCPMMP1_06_somatosensory_motor-lh', 'rh.HCPMMP1_06_somatosensory_motor-rh',),
         'HCPMMP1_paracentral_lob_mid_cingulate':   ('lh.HCPMMP1_07_paracentral_lob_mid_cingulate-lh', 'rh.HCPMMP1_07_paracentral_lob_mid_cingulate-rh',),
         'HCPMMP1_premotor':                        ('lh.HCPMMP1_08_premotor-lh', 'rh.HCPMMP1_08_premotor-rh',),
         'HCPMMP1_opercular_posterior':             ('lh.HCPMMP1_09_opercular_posterior-lh', 'rh.HCPMMP1_09_opercular_posterior-rh',),
         'HCPMMP1_auditory_primary':                ('lh.HCPMMP1_10_auditory_primary-lh', 'rh.HCPMMP1_10_auditory_primary-rh',),
         'HCPMMP1_auditory_association':            ('lh.HCPMMP1_11_auditory_association-lh', 'rh.HCPMMP1_11_auditory_association-rh',),
         'HCPMMP1_insular_frontal_opercular':       ('lh.HCPMMP1_12_insular_frontal_opercular-lh', 'rh.HCPMMP1_12_insular_frontal_opercular-rh',),
         'HCPMMP1_temporal_medial':                 ('lh.HCPMMP1_13_temporal_medial-lh', 'rh.HCPMMP1_13_temporal_medial-rh',),
         'HCPMMP1_lateral_temporal':                ('lh.HCPMMP1_14_lateral_temporal-lh', 'rh.HCPMMP1_14_lateral_temporal-rh',),
         'HCPMMP1_temp_par_occ_junc':               ('lh.HCPMMP1_15_temporal_parietal_occipital_junction-lh', 'rh.HCPMMP1_15_temporal_parietal_occipital_junction-rh',),
         'HCPMMP1_parietal_superior':               ('lh.HCPMMP1_16_parietal_superior-lh', 'rh.HCPMMP1_16_parietal_superior-rh',),
         'HCPMMP1_parietal_inferior':               ('lh.HCPMMP1_17_parietal_inferior-lh', 'rh.HCPMMP1_17_parietal_inferior-rh',),
         'HCPMMP1_cingulate_posterior':             ('lh.HCPMMP1_18_cingulate_posterior-lh', 'rh.HCPMMP1_18_cingulate_posterior-rh',),
         'HCPMMP1_cingulate_anterior':              ('lh.HCPMMP1_19_cingulate_anterior_prefrontal_medial-lh', 'rh.HCPMMP1_19_cingulate_anterior_prefrontal_medial-rh',),
         'HCPMMP1_frontal_orbital_polar':           ('lh.HCPMMP1_20_frontal_orbital_polar-lh', 'rh.HCPMMP1_20_frontal_orbital_polar-rh',),
         'HCPMMP1_frontal_inferior':                ('lh.HCPMMP1_21_frontal_inferior-lh', 'rh.HCPMMP1_21_frontal_inferior-rh',),
         'HCPMMP1_prefrontal_dorsolateral':         ('lh.HCPMMP1_22_prefrontal_dorsolateral-lh', 'rh.HCPMMP1_22_prefrontal_dorsolateral-rh',),
    }

    jwg_clusters = {
         'JWG_aIPS':                                ('lh.JWG_lat_aIPS-lh', 'rh.JWG_lat_aIPS-rh',),
         'JWG_IPS_PCeS':                            ('lh.JWG_lat_IPS_PCeS-lh', 'rh.JWG_lat_IPS_PCeS-rh',),
         'JWG_M1':                                  ('lh.JWG_lat_M1-lh', 'rh.JWG_lat_M1-rh',),
    }

    # all_clusters = {**visual_field_clusters, **glasser_clusters, **jwg_clusters}
    all_clusters = dict(visual_field_clusters.items() + glasser_clusters.items() + jwg_clusters.items())
    areas = [item for sublist in [all_clusters[k] for k in all_clusters.keys()] for item in sublist]
    print(areas)

    return all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters
