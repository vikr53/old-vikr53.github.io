from gurobipy import *
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = patches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

if __name__ == "__main__":
	try:
		# Create a new model
		m = Model("footstep")

		# Create variables
		# Create N footstep variables (x,y,theta,s,c)
		N = 18
		footsteps = [m.addVars(5,lb=-5,name="F"+str(i)) for i in range(0,N)]

		# Trig approx functions
		S = [m.addVars(5,vtype=GRB.BINARY, name="S"+str(i)) for i in range(0,N)]
		C = [m.addVars(5,vtype=GRB.BINARY, name="C"+str(i)) for i in range(0,N)]

		# Safe Regions
		N_REG = 5
		H = [m.addVars(N_REG,vtype=GRB.BINARY, name="H"+str(i)) for i in range(0,N)]

		# Set constraints

		# DEFINE REGIONS
		## SCENARIO 1
		# R1_xmax = 1
		# R1_xmin = 0
		# R1_ymax = 1
		# R1_ymin = 0
		# R1_midpt = [(R1_xmax + R1_xmin)/2 , (R1_ymax + R1_ymin)/2]

		# R2_xmax = 1.6
		# R2_xmin = 1.1
		# R2_ymax = 0.85
		# R2_ymin = -0.5
		# R2_midpt = [(R2_xmax + R2_xmin)/2 , (R2_ymax + R2_ymin)/2]

		# R3_xmax = 2.2
		# R3_xmin = 1.65
		# R3_ymax = 1.8
		# R3_ymin = 0
		# R3_midpt = [(R3_xmax + R3_xmin)/2 , (R3_ymax + R3_ymin)/2]

		# R4_xmax = 2.35
		# R4_xmin = 1.2
		# R4_ymax = 2.5
		# R4_ymin = 1.85
		# R4_midpt = [(R4_xmax + R4_xmin)/2 , (R4_ymax + R4_ymin)/2]

		# R5_xmax = 1
		# R5_xmin = -0.5
		# R5_ymax = 2
		# R5_ymin = 1.1
		# R5_midpt = [(R5_xmax + R5_xmin)/2 , (R5_ymax + R5_ymin)/2]

		## SCENARIO 2
		R1_xmax = 1
		R1_xmin = 0
		R1_ymax = 1
		R1_ymin = 0
		R1_midpt = [(R1_xmax + R1_xmin)/2 , (R1_ymax + R1_ymin)/2]

		R2_xmax = 1.6
		R2_xmin = 1.1
		R2_ymax = 2
		R2_ymin = 0
		R2_midpt = [(R2_xmax + R2_xmin)/2 , (R2_ymax + R2_ymin)/2]

		R3_xmax = 2
		R3_xmin = 1.1
		R3_ymax = 2.5
		R3_ymin = 2.1
		R3_midpt = [(R3_xmax + R3_xmin)/2 , (R3_ymax + R3_ymin)/2]

		R4_xmax = 1
		R4_xmin = -0.5
		R4_ymax = 2.7
		R4_ymin = 2.1
		R4_midpt = [(R4_xmax + R4_xmin)/2 , (R4_ymax + R4_ymin)/2]

		R5_xmax = 2
		R5_xmin = 1.5
		R5_ymax = 3
		R5_ymin = 2.55
		R5_midpt = [(R5_xmax + R5_xmin)/2 , (R5_ymax + R5_ymin)/2]

		# R6_xmax = 0
		# R6_xmin = -1
		# R6_ymax = 4
		# R6_ymin = 3.1

		# R7_xmax = 2.5
		# R7_xmin = 1
		# R7_ymax = 3.5
		# R7_ymin = 2.6

		# R8_xmax = 2.5
		# R8_xmin = 2
		# R8_ymax = 2.6
		# R8_ymin = 1.5


		A_1 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_1 = [R1_xmax,-R1_xmin,R1_ymax,-R1_ymin,math.pi,math.pi/2]
		print(A_1[1][2])
		
		A_2 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_2 = [R2_xmax,-R2_xmin,R2_ymax,-R2_ymin,math.pi,math.pi/2]
		
		A_3 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_3 = [R3_xmax,-R3_xmin,R3_ymax,-R3_ymin,math.pi,math.pi/2]

		A_4 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_4 = [R4_xmax,-R4_xmin,R4_ymax,-R4_ymin,math.pi,math.pi/2]

		A_5 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_5 = [R5_xmax,-R5_xmin,R5_ymax,-R5_ymin,math.pi,math.pi/2]

		# A_6 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_6 = [R6_xmax,-R6_xmin,R6_ymax,-R6_ymin,math.pi,math.pi/2]

		# A_7 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_7 = [R7_xmax,-R7_xmin,R7_ymax,-R7_ymin,math.pi,math.pi/2]

		# A_8 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_8 = [R8_xmax,-R8_xmin,R8_ymax,-R8_ymin,math.pi,math.pi/2]

		# All footsteps must be in the regions
		for c in range(0,N):
			for i in range(0,len(A_1)):
				M = 1000
				# Region 1
				m.addConstr(-M*(1-H[c][0]) + quicksum(A_1[i][j]*footsteps[c][j] for j in range(0,3)) - b_1[i] <= 0)
				# Region 2
				m.addConstr(-M*(1-H[c][1]) + quicksum(A_2[i][j]*footsteps[c][j] for j in range(0,3)) - b_2[i] <= 0)
				# Region 3
				m.addConstr(-M*(1-H[c][2]) + quicksum(A_3[i][j]*footsteps[c][j] for j in range(0,3)) - b_3[i] <= 0)
				# Region 4
				m.addConstr(-M*(1-H[c][3]) + quicksum(A_4[i][j]*footsteps[c][j] for j in range(0,3)) - b_4[i] <= 0)
				# Region 5
				m.addConstr(-M*(1-H[c][4]) + quicksum(A_5[i][j]*footsteps[c][j] for j in range(0,3)) - b_5[i] <= 0)
				# # Region 6
				# m.addConstr(-M*(1-H[c][5]) + quicksum(A_6[i][j]*footsteps[c][j] for j in range(0,3)) - b_6[i] <= 0)
				# # Region 7
				# m.addConstr(-M*(1-H[c][6]) + quicksum(A_7[i][j]*footsteps[c][j] for j in range(0,3)) - b_7[i] <= 0)
				# # Region 8
				# m.addConstr(-M*(1-H[c][7]) + quicksum(A_8[i][j]*footsteps[c][j] for j in range(0,3)) - b_8[i] <= 0)

			# Constraint that the sum of H must be 1 for every foothold
			m.addConstr(quicksum(H[c][j] for j in range(0,N_REG)) == 1 )

		#Reachability constraint
		for c in range(2,N):
			# if odd after f1,f2 (fixed), so f3, f5, f7, ...
			# Let's say odd is finding a step for right leg
			if (c % 2 != 0):
				p1 = [0,0]
				p2 = [0,-0.7]
				d1 = 0.4
				d2 = 0.4
				xn = footsteps[c][0]
				yn = footsteps[c][1]
				xc = footsteps[c-1][0]
				yc = footsteps[c-1][1]
				thetac = footsteps[c-1][2]
				term1_a = xn - (xc + p1[0]*footsteps[c-1][4] - p1[1]*footsteps[c-1][3])
				term2_a = yn - (yc + p1[0]*footsteps[c-1][3] + p1[1]*footsteps[c-1][4])
				# term1_a = xn - (xc + p1[0])
				# term2_a = yn - (yc + p1[1])
				m.addQConstr(term1_a*term1_a + term2_a*term2_a <= d1*d1)
				m.addQConstr(term1_a*term1_a + term2_a*term2_a <= d1*d1)

				# term1_b = xn - (xc + p2[0])
				# term2_b = yn - (yc + p2[1])
				term1_b = xn - (xc + p2[0]*footsteps[c-1][4] - p2[1]*footsteps[c-1][3])
				term2_b = yn - (yc + p2[0]*footsteps[c-1][3] + p2[1]*footsteps[c-1][4])
				m.addQConstr(term1_b*term1_b + term2_b*term2_b <= d2*d2)
			else:
				# finding step for left leg
				print("OTHER LEG")
				p1 = [0, -0.1]
				p2 = [0,0.8]
				d1 = 0.55
				d2 = 0.55
				xn = footsteps[c][0]
				yn = footsteps[c][1]
				xc = footsteps[c-1][0]
				yc = footsteps[c-1][1]
				thetac = footsteps[c-1][2]
				term1 = xn - (xc + p1[0]*footsteps[c-1][4] - p1[1]*footsteps[c-1][3])
				term2 = yn - (yc + p1[0]*footsteps[c-1][3] + p1[1]*footsteps[c-1][4])
				# term1 = xn - (xc + p1[0])
				# term2 = yn - (yc + p1[1])
				m.addQConstr(term1*term1 + term2*term2 <= d1*d1)

				# term1 = xn - (xc + p2[0])
				# term2 = yn - (yc + p2[1])
				term1 = xn - (xc + p2[0]*footsteps[c-1][4] - p2[1]*footsteps[c-1][3])
				term2 = yn - (yc + p2[0]*footsteps[c-1][3] + p2[1]*footsteps[c-1][4])
				m.addQConstr(term1*term1 + term2*term2 <= d2*d2)

		# Add constraints for sin
		for c in range(0,N):
			for i in range(0,5):
				M = 1000
				if i == 0:
					phi_l = -math.pi
					phi_lp1 = 1-math.pi
					g_l = -1
					h_l = -math.pi
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 1:
					phi_l = 1-math.pi
					phi_lp1 = -1
					g_l = 0
					h_l = -1
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 2:
					phi_l = -1
					phi_lp1 = 1
					g_l = 1
					h_l = 0
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 3:
					phi_l = 1
					phi_lp1 = math.pi-1
					g_l = 0
					h_l = 1
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 4:
					phi_l = math.pi-1
					phi_lp1 = math.pi
					g_l = -1
					h_l = math.pi
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
			m.addConstr(quicksum(S[c][j] for j in range(0,5)) == 1 )

		# Add constraints for cos
		for c in range(0,N):
			for i in range(0,5):
				M = 1000
				if i == 0:
					phi_l = -math.pi
					phi_lp1 = -1-math.pi/2
					g_l = 0
					h_l = -1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 1:
					phi_l = -1-math.pi/2
					phi_lp1 = 1-math.pi/2
					g_l = 1
					h_l = math.pi/2
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 2:
					phi_l = 1-math.pi/2
					phi_lp1 = math.pi/2-1
					g_l = 0
					h_l = 1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 3:
					phi_l = math.pi/2-1
					phi_lp1 = math.pi/2+1
					g_l = -1
					h_l = math.pi/2
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 4:
					phi_l = math.pi/2+1
					phi_lp1 = math.pi
					g_l = 0
					h_l = -1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
			m.addConstr(quicksum(C[c][j] for j in range(0,5)) == 1 )

		# # Set first two footholds
		init_theta = 0
		f1_s = math.sin(init_theta)
		f1_c = math.cos(init_theta)
		m.addConstr(footsteps[0][0] == 0)
		m.addConstr(footsteps[0][1] == 0.6)
		m.addConstr(footsteps[0][3] == f1_s)
		m.addConstr(footsteps[0][4] == f1_c)
		m.addConstr(S[0][2] == 1)
		m.addConstr(C[0][2] == 1)
		
		m.addConstr(footsteps[1][0] == 0)
		m.addConstr(footsteps[1][1] == 0)
		m.addConstr(footsteps[1][2] == init_theta)
		m.addConstr(footsteps[1][3] == f1_s)
		m.addConstr(footsteps[1][4] == f1_c)
		m.addConstr(S[1][2] == 1)
		m.addConstr(C[1][2] == 1)

		# Add constraint of how much foot can rotate in one step
		for c in range(1,N):
			del_theta_max = math.pi/8
			m.addConstr((footsteps[c][2] - footsteps[c-1][2]) <= del_theta_max)
			m.addConstr( (footsteps[c][2] - footsteps[c-1][2]) >= -del_theta_max)

		#########################################################
		####################### SPECS ###########################
		#########################################################

		########## SCENARIO 1 ##############
		# Visit region 3 or 4 eventually
		#m.addConstr(quicksum(H[j][2] for j in range(0,N)) + quicksum(H[j][3] for j in range(0,N)) >= 1)


		########## SCENARIO 2 ############### 
		# # Until
		# # Always be in regions 2 or 3 before entering region 4
		# reg1 = 0 # region 2
		# reg2 = 1 # region 3
		# phi2_reg = 2 # region 4

		# T = [m.addVar(vtype=GRB.BINARY, name="T"+str(i)) for i in range(0,N)]

		# # Base case
		# m.addConstr(T[N-1] == H[N-1][phi2_reg])

		# # Satisfiability constraint
		# # m.addConstr(quicksum(T[j] for j in range(0,N)) == N)
		# m.addConstr(T[0] == 1)

		# Pphi1 = [m.addVar(vtype=GRB.BINARY, name="Pphi1"+str(i)) for i in range(0,N-1)]
		# B = [m.addVar(vtype=GRB.BINARY, name="B"+str(i)) for i in range(0,N-1)]

		# # Recursive constraints
		# for i in range(0,N-1):
		# 	M = 1000
		# 	delta = 0.001
			
		# 	m.addConstr(H[i][reg1] + H[i][reg2] - 1 >= -M*(1-Pphi1[i]))
		# 	m.addConstr(H[i][reg1] + H[i][reg2] - 1 + delta <=  M*(Pphi1[i]))

		# 	# Term in parenthesis
		# 	m.addConstr(Pphi1[i] + T[i+1] - 2 >= -M*(1-B[i]))
		# 	m.addConstr(Pphi1[i] + T[i+1] - 2 + delta <= M*(B[i]))

		# 	# Final constraint
		# 	m.addConstr(H[i][phi2_reg] + B[i] - 1 >= -M*(1-T[i]))
		# 	m.addConstr(H[i][phi2_reg] + B[i] - 1 + delta <= M*(T[i]))

		########## SCENARIO 3 ##############
		ni = 0
		nf = 15
		m.addConstr(quicksum(H[i][1] for i in range(ni-1,nf)) >= nf-ni+1)

		# Set objective
		g = [1.5,2.2,3*math.pi/4]
		e0 = footsteps[N-1][0]-g[0] 
		e1 = footsteps[N-1][1]-g[1] 
		e2 = footsteps[N-1][2]-g[2] 
		Q = [[300,0,0],[0,300,0],[0,0,300]]

		term_cost = (e0)*(e0)*Q[0][0] + (e0)*(e1)*Q[1][0] + (e2)*(e0)*Q[2][0] + (e0)*(e1)*Q[0][1]\
			+(e1)*(e1)*Q[1][1]+(e2)*(e1)*Q[1][2]+(e0)*(e2)*Q[2][0]+(e1)*(e2)*Q[2][1]+(e2)*(e2)*Q[2][2]

		# Calculate incremental costs
		R = [[1,0,0],[0,1,0],[0,0,1]]
		inc_cost = quicksum((footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][0]-footsteps[j-1][0])*R[0][0] + (footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][1]-footsteps[j-1][1])*R[1][0]\
			+(footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][0]-footsteps[j-1][0])*R[2][0] + (footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][1]-footsteps[j-1][1])*R[0][1]\
			+(footsteps[j][1]-footsteps[j-1][1])*(footsteps[j][1]-footsteps[j-1][1])*R[1][1] + (footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][1]-footsteps[j-1][1])*R[1][2]\
			+(footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][2]-footsteps[j-1][2])*R[2][0] + (footsteps[j][1]-footsteps[j-1][1])*(footsteps[j][2]-footsteps[j-1][2])*R[2][1]\
			+(footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][2]-footsteps[j-1][2])*R[2][2] for j in range(0,N))

		#inc_cost = quicksum((footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][0]-footsteps[j-1][0])*R[0][0] for j in range(0,N))
		#inc_cost = 0
		m.setObjective( term_cost + inc_cost
				\
				, GRB.MINIMIZE )

		#print(f.values())
		m.optimize()

		footsteps_x = []
		footsteps_y = []
		footsteps_theta = []

		# Get x values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[0]")
			footsteps_x.append(v.X)

		# Get y values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[1]")
			footsteps_y.append(v.X)

		# Get theta values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[2]")
			footsteps_theta.append(v.X)

		print(footsteps_x)
		print(footsteps_y)
		print(footsteps_theta)

		for v in m.getVars():
			print('%s %g' % (v.varName,v.x))

		###### PLOT ######

		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		ax1.set(xlim=(-2,5), ylim=(-2,5))

		# Plot initial foot stance
		ax1.plot(footsteps_x[0],footsteps_y[0], 'bo')
		# ax1.text(footsteps_x[0]-0.05,footsteps_y[0]+0.02, str(1), fontsize=8, color='blue' )
		ax1.plot(footsteps_x[1],footsteps_y[1], 'r*')
		#ax1.text(footsteps_x[1],footsteps_x[1]-0.02, str(2), fontsize=8, color='red' )
		ax1.arrow(footsteps_x[0],footsteps_y[0],0.25*math.cos(footsteps_theta[0]),0.25*math.sin(footsteps_theta[0]))
		ax1.arrow(footsteps_x[1],footsteps_y[1],0.25*math.cos(footsteps_theta[1]),0.25*math.sin(footsteps_theta[1]))

		# Plot safe region 1
		rect = patches.Rectangle((R1_xmin,R1_ymin),R1_xmax-R1_xmin,R1_ymax-R1_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 2
		rect = patches.Rectangle((R2_xmin,R2_ymin),R2_xmax-R2_xmin,R2_ymax-R2_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 3
		rect = patches.Rectangle((R3_xmin,R3_ymin),R3_xmax-R3_xmin,R3_ymax-R3_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 4
		rect = patches.Rectangle((R4_xmin,R4_ymin),R4_xmax-R4_xmin,R4_ymax-R4_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 5
		rect = patches.Rectangle((R5_xmin,R5_ymin),R5_xmax-R5_xmin,R5_ymax-R5_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# # Plot safe region 6
		# rect = patches.Rectangle((R6_xmin,R6_ymin),R6_xmax-R6_xmin,R6_ymax-R6_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)
		# # Plot safe region 7
		# rect = patches.Rectangle((R7_xmin,R7_ymin),R7_xmax-R7_xmin,R7_ymax-R7_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)
		# # Plot safe region 8
		# rect = patches.Rectangle((R8_xmin,R8_ymin),R8_xmax-R8_xmin,R8_ymax-R8_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)

		def animate(i):
			if (i % 2 == 0) & (i < len(footsteps_x)-2):
				# It is a left footstep
				cur_x = footsteps_x[i+2]
				cur_y = footsteps_y[i+2]
				cur_theta = footsteps_theta[i+2]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				#bl_x = cur_x
				#bl_y = cur_y
				ax1.plot(cur_x,cur_y,'bo')
				# ax1.text(cur_x-0.18,cur_y-0.1, str(i+3), fontsize=8, color='blue' )
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))

				p1 = [0,0.1]
				p2 = [0,-0.8]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

				# circ1 = patches.Circle((center_x1,center_y1),0.55,linewidth=1, edgecolor='r',facecolor='none')
				# circ2 = patches.Circle((center_x2,center_y2),0.55,linewidth=1, edgecolor='r',facecolor='none')
				# ax1.add_patch(circ1)
				# ax1.add_patch(circ2)
				
			elif (i % 2 != 0) & (i < len(footsteps_x)-2):
				cur_x = footsteps_x[i+2]
				cur_y = footsteps_y[i+2]
				cur_theta = footsteps_theta[i+2]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				# It is a right footstep
				ax1.plot(cur_x,cur_y,'r*')
				#ax1.text(cur_x-0.18,cur_y-0.1, str(i+3), fontsize=8, color='red' )
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				arrow = ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))
				p1 = [0,-0.1]
				p2 = [0,0.8]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

		ani = animation.FuncAnimation(fig, animate, interval=1000)
		ax1.legend(["Right foot", "Left foot"])
		offset = 0.1
		ax1.text(R1_midpt[0]-offset,R1_midpt[1]-offset,"R1")
		ax1.text(R2_midpt[0]-offset,R2_midpt[1]-offset,"R2")
		ax1.text(R3_midpt[0]-offset,R3_midpt[1]-offset,"R3")
		ax1.text(R4_midpt[0]-offset,R4_midpt[1]-offset,"R4")
		ax1.text(R5_midpt[0]-offset,R5_midpt[1]-offset,"R5")

		plt.show()

	except GurobiError as e:
		print('Error code' + str(e.errno)+":"+str(e))