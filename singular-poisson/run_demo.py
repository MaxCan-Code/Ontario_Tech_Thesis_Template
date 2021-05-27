import demo_singular_poisson as dem
initial_cells_lst = [8 * 2 ** i for i in range(2, 4)]
R_lst = [i * 25 for i in range(2, 5)]
refn_lst = range(1, 3)
params_d = {'initial cells': initial_cells_lst, 'R': R_lst, 'refinements': refn_lst}
dem.sv_conv_table(params_d)
