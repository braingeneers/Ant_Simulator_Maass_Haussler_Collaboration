import numpy as np

CART_POLE = 0

task_to_tasktype = {'cartpole': 'control', 'halfcheetah': 'control',
                    'ant': 'control', 'walker': 'control',
                    'const': 'function_approximation',
                    'identity': 'function_approximation', 'linear': 'function_approximation',
                    'sin': 'function_approximation', 'poly': 'function_approximation',
                    'spike_interval_classification': 'spike_classification',
                    "spike_train_classification": "spike_classification",
                    'correlation': 'correlation',
                    'gratings': 'gratings',
                    'contrast': 'contrast'}

task_name_env_map = {
    'cartpole': 'CartPole-v0',
    # 'halfcheetah': 'HalfCheetahPyBulletEnv-v0',
    'halfcheetah': 'HalfCheetahMuJoCoEnv-v0',
    'ant': 'AntMuJoCoEnv-v0',
    # 'ant': 'AntPyBulletEnv-v0',
    'walker': 'BipedalWalker-v3'
}

sample_neuron_params = {
    'V_th': -40.63269100412709,
    'g': 4.339669129057476,
    'E_L': -63.7881507873535,
    'C_m': 278.523421091562,
    't_ref': 7.3500000000000005,
    'V_reset': -63.7881507873535,
    'asc_init': [0., 0.],
    'k': [0.003, 0.03],
    'asc_amps': [-19.13204504, -94.15816779],
    'V_dynamics_method': 'linear_exact',
    'tau_syn': [5.5, 8.5, 2.8, 5.8]
}

E_neuron_params = {
    'V_th': -48.11438,
    'g': 1.70498466213,
    'E_L': -75.94854,
    'C_m': 65.24113,
    't_ref': 7.35,
    'V_reset': -75.94854,
    'e_i': 1.,
    'k': np.array([[0.01, 0.1]]),
    'neuron_type_ids': '501848315',
    'neuron_type_names': 'e4Rorb',
    'neuron_type_param_files': '501848315_glif_lif_asc_psc.json',
    'tau_syn': np.array([[5.5, 8.5, 2.8, 5.8]])
}

I_neuron_params = {
    'V_th': -49.639347,
    'g': 3.92054555,
    'E_L': -79.0417277018229,
    'C_m': 60.72689987399939,
    't_ref': 1.45,
    'V_reset': -79.0417277018229,
    'e_i': -1.,
    'k': np.array([[0.03, 0.3]]),
    'neuron_type_ids': '313862167',
    'neuron_type_names': 'i4Sst',
    'neuron_type_param_files': '313862167_glif_lif_asc_psc.json',
    'tau_syn': np.array([[5.5, 8.5, 2.8, 5.8]])
}


for k, v in sample_neuron_params.items():
    sample_neuron_params[k] = np.array([v])

sorted_neuron_type_ids= ['478412623', '318556138', '482764620', '474267418', '478949560', '475622680'
                         '480952106', '475517090', '482690728', '474626527', '327962063', '478058328'
                         '464198958', '479219923', '478110866', '328549885', '476104386', '336676216'
                         '478396248', '324266189', '515305346', '471087830', '484659182', '476686112'
                         '504608602', '487667205', '478793814', '313861608', '313862167', '475585413'
                         '488501071', '481093525', '501848315', '484679812', '478107198', '473564515'
                         '487176969', '314900022', '314642645', '477135941', '341442651', '324493977'
                         '483018019', '386970660', '479179020', '475622793', '486146828', '476131588'
                         '473020156', '480122859', '318733871', '480090260', '479492633', '480087928'
                         '482809953', '395830185', '487664663', '478958894', '480351780', '476457450'
                         '469793303', '324025297', '471789504', '476056333', '479225080', '487601493'
                         '318808427', '476451456', '479220013', '386049446', '487661754', '488419491'
                         '517645929', '466632464', '480124551', '422738880', '485880739', '476263004'
                         '509003464', '484635029', '486052980', '486176465', '324025371', '473943881'
                         '501282204', '370351753', '486110216', '320207387', '518271679', '484737013'
                         '464188580', '517982558', '485836906', '485574832', '471819401', '486146717'
                         '480169178', '518290966', '481017021', '507996688', '480353286', '539742766'
                         '482773968', '476218657', '490916919', '354190013', '469753383', '468120757'
                         '476048909', '471129934', '478828646']

sorted_neuron_type_indices = [ 81,  57,  46,  32,  59,  31,  55,  84,  82,  66,  27,  12,  83,  25,  45, 103, 102,  53,
                               80,  63,  11,  54,  98,  65, 105,  58,  60,  62,  68,  78,  22,  14,  41,  43, 24,  79,
                               104,  18,  21,  36,  49,  64,  50,  77,  28,  44,  52,  86,  35,  40,  74,  90,  94,  42,
                               70,   5,  16,  61,  85,  72,  99,  87, 110,  17,  30,  39,  51,  69,  73,  71,   0,  10,
                               97, 109,  34,   3,  23,  75, 100, 108,   2,   8,  33,  37,  67,   7,  26,  76,  91,  92,
                               95,   6,  15,  38, 106,  48,  88,  20,  56,   4,  89,  47, 107,   9,  19,  29,  96,  93,
                               101,   1,  13]

e_class_priorities = ['e23Cux2', 'e4Rorb', 'e5noRbp4', 'e4other', 'e5Rbp4', 'e4Scnn1a', 'e4Nr5a1', 'e6Ntsr1']

i_class_priorities = ['i23Sst','i4Sst', 'i5Pvalb', 'i23Pvalb', 'i4Pvalb', 'i5Sst', 'i23Htr3a', 'i4Htr3a', 'i5Htr3a',
                      'i1Htr3a', 'i6Pvalb', 'i6Sst', 'i6Htr3a']


#  observation indices

'''
BiPedalWalker:
        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible. idx: 0
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too) idx: 5
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0, # idx: 7
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0, # idx: 9
            self.joints[2].angle, # idx: 10
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,  # idx: 12
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0  # idx: 14
            ]
'''
obs_indices = {
    # height of torso and four angles
    # 'AntMuJoCoEnv-v0': [0, 5, 6, 7, 8, 9, 10, 11, 12],
    'AntMuJoCoEnv-v0': [0, 5, 6, 7, 8, 9, 10, 11, 12],
    'AntPyBulletEnv-v0': [0, 8, 10, 12, 14, 16, 18, 20, 22],
    'BipedalWalker-v3': [0, 4, 6, 8, 9, 11, 13],
    'CartPole-v0': [0, 1, 2, 3]
}

target_n_synapses = 8  # nice number :)

### ANT PyBullet https://github.com/openai/gym/issues/585:
"""
Pos (Torso)
0 x
1 y   # my guess: idx 0!
2 z
Orient (Torso)
3 x
4 y
5 z
6 w
joint angles
7 1 Front left leg hip angle <---  # my guess: idx 8
8 2 Front left leg ankle angle <---  # my guess: idx 10
9 3 Back left leg hip angle <---       # 12
10 4 Back left leg ankle angle <---  # my guess: idx 14
11 5 Back left leg hip angle <---      # 16
12 6 Back right leg ankle angle <---  # my guess: idx 18
13 7 Front right leg hip angle <---    # 20
14 8 Front right leg ankle angle <--- # my guess: idx 22

Vel (Torso)
15 x
16 y
17 z

Angular Vel (Torso)
18 x
19 y
20 z

joint vel
21 1 Front left leg hip angle <---
22 2 Front left leg ankle angle <---
23 3 Back left leg hip angle <---
24 4 Back left leg ankle angle <---
25 5 Back right leg hip angle <---
26 6 Back right leg ankle angle <---
27 7 Front right leg hip angle <---
28 8 Front right leg ankle angle <---
"""