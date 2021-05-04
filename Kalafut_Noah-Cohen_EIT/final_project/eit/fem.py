import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate


class Mesh:
	def __init__(self, points, triangles, bdy_idx, vol_idx):
		# self.p	array with the node points (sorted)
		#		   type : np.array dim: (n_p, 2)
		# self.n_p  number of node points
		#		   type : int
		# self.t	array with indices of points per segment
		#		   type : np.array dim: (n_s, 3)
		# self.n_t  number of triangles
		#		   type : int
		# self.bdy_idx  array with the indices of boundary points
		#		   type : np.array dim: (n_b)
		# self.vol_idx  array with the indices of interior points
		#		   type : np.array dim: (n_i)

		self.p = points
		self.t = triangles

		self.n_p = self.p.shape[0]
		self.n_t = self.t.shape[0]

		self.bdy_idx = bdy_idx
		self.vol_idx = vol_idx


class V_h:
	def __init__(self, mesh):
		# self.mesh Mesh object containg geometric info type: Mesh
		# self.dim  dimension of the space			  type: in

		self.mesh = mesh
		self.dim = mesh.n_p


def stiffness_matrix(v_h, sigma_vec):
	''' S = stiffness_matrix(v_h, sigma_vec)
		function to assemble the stiffness matrix 
		for the Poisson equation 
		input: v_h: this contains the information 
			   approximation space. For simplicity
			   we suppose that the space is piece-wise
			   linear polynomials
			   sigma_vec: values of sigma at each 
			   triangle
	'''
	# define a local handles 
	t = v_h.mesh.t
	p = v_h.mesh.p

	# we define the arrays for the indicies and the values 
	idx_i = np.zeros((v_h.mesh.n_t, 9), dtype = np.int)
	idx_j = np.zeros((v_h.mesh.n_t, 9), dtype = np.int)
	vals = np.zeros((v_h.mesh.n_t, 9), dtype = np.float64)

	# Assembly the matrix
	for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
		# row of t = node numbers of the 3 corners of triangle e
		nodes = t[e,:]
  
		# compute S_local here
		# Calculate |K|
		pos = p[nodes]
		K = pos[0,0]*(pos[1,1]-pos[2,1])+pos[1,0]*(pos[2,1]-pos[0,1])\
			+pos[2,0]*(pos[0,1]-pos[1,1])
		K = 12*np.abs(K)
		
		# Calculate gradients
		a = np.zeros(3);b = np.zeros(3);c = np.zeros(3)
		for i in range(3):
			# Define j,k
			j=i+1;k=i+2;j=j%3;k=k%3
			x=0;y=1
			a[i] = (pos[j,x]*pos[k,y]-pos[j,y]*pos[k,x])/(2*K)
			b[i] = (pos[j,y]-pos[k,y])/(2*K)
			c[i] = (pos[k,x]-pos[j,x])/(2*K)
		
		# Form base matrix
		S_local = np.outer(b,b)+np.outer(c,c)
		
		# Approximate sigma at the centroid
		# sigma = np.sum(sigma_vec[nodes])/3
		# If sigma at the centroid is provided, we can instead use
		sigma = sigma_vec[e]
		
		# Compute final S_local
		S_local = sigma * S_local * K
		
		# add S_local  to 9 entries of global K
		idx_i[e,:] = (np.ones((3,1))*nodes).T.reshape((9,))
		idx_j[e,:] = (np.ones((3,1))*nodes).reshape((9,))
		# add the values of the matrix computed in S local here
		vals[e,:] = S_local.reshape((9,))

	# we add all the indices to make the matrix
	S_coo = spsp.coo_matrix((vals.reshape((-1,)), 
							(idx_i.reshape((-1,)), 
							 idx_j.reshape((-1,)))), shape=(v_h.dim, v_h.dim))

	return spsp.lil_matrix(S_coo) 


#####################################################
def mass_matrix(v_h):
	''' M = mass_matrix(v_h)
		function to assemble the mass matrix 
		for the Poisson equation 
		input: v_h: this contains the information 
			   approximation space. For simplicity
			   we suppose that the space is piece-wise
			   linear polynomials
	'''

	# define a local handles 
	t = v_h.mesh.t
	p = v_h.mesh.p

	idx_i = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
	idx_j = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
	vals = np.zeros((v_h.mesh.n_t, 9), dtype  = np.float64)

	# local mass matrix (so we don't need to compute it at each iteration)
	MK = 1/12*np.array([ [2., 1., 1.], 
						 [1., 2., 1.],
						 [1., 1., 2.]])


	# Assembly the matrix
	for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
		# row of t = node numbers of the 3 corners of triangle e
		nodes = t[e,:]
  
		# compute the area of each element
		pos = p[nodes]
		K = pos[0,0]*(pos[1,1]-pos[2,1])+pos[1,0]*(pos[2,1]-pos[0,1])\
			+pos[2,0]*(pos[0,1]-pos[1,1])
		K = 12*np.abs(K)
		Area = K

		# then compute the local area matrix 
		M_local = Area*MK
		
		# add M_local  to 9 entries of global K
		idx_i[e,:] = (np.ones((3,1))*nodes).T.reshape((9,))
		idx_j[e,:] = (np.ones((3,1))*nodes).reshape((9,))

		# uncomment here once you computed M_local
		vals[e,:] = M_local.reshape((9,))

	# we add all the indices to make the matrix
	M_coo = spsp.coo_matrix((vals.reshape((-1,)), 
							(idx_i.reshape((-1,)), 
							 idx_j.reshape((-1,)))), shape=(v_h.dim, v_h.dim))

	return spsp.lil_matrix(M_coo) 


def projection_v_w(v_h):
	''' M = mass_matrix(v_h)
		function to assemble the mass matrix 
		for the Poisson equation 
		input: v_h: this contains the information 
			   approximation space. For simplicity
			   we suppose that the space is piece-wise
			   linear polynomials
	'''

	# define a local handles 
	t = v_h.mesh.t
	p = v_h.mesh.p

	idx_i = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
	idx_j = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
	vals = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)

	# Assembly the matrix
	for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
		# row of t = node numbers of the 3 corners of triangle e
		nodes = t[e,:]
  
		# compute the area

		# add S_local  to 9 entries of global K
		idx_i[e,:] = nodes
		idx_j[e,:] = e*np.ones((3,))
		# uncomment and compute this one
		# vals[e,:] = ...

	# we add all the indices to make the matrix
	M_coo = spsp.coo_matrix((vals.reshape((-1,)), 
							(idx_i.reshape((-1,)), 
							 idx_j.reshape((-1,)))), 
							shape=(v_h.dim, v_h.mesh.n_t))

	return spsp.lil_matrix(M_coo) 


def partial_deriv_matrix(v_h):
	''' Dx, Dy, Surf = mass_matrix(v_h)
		function to assemble the mass matrix 
		for the Poisson equation 
		input: v_h: this contains the information 
			   approximation space. For simplicity
			   we suppose that the space is piece-wise
			   linear polynomials
		output: Dx matrix to compute weak derivatives
				Dx matrix to compute weak derivative
				M_t mass matrix in W (piece-wise constant matrices)
	'''
	# define a local handles 
	t = v_h.mesh.t
	p = v_h.mesh.p

	# number of triangles
	n_t = v_h.mesh.n_t

	# allocating the indices
	idx_i = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
	idx_j = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
	vals_x = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)
	vals_y = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)
	vals_s = np.zeros((v_h.mesh.n_t, 1), dtype  = np.float64)

	# Assembly the matrix
	for e in range(n_t):  #
		nodes = t[e,:]
  
		# compute partial derivatives and area 
		# Calculate |K|
		pos = p[nodes]
		K = pos[0,0]*(pos[1,1]-pos[2,1])+pos[1,0]*(pos[2,1]-pos[0,1])\
			+pos[2,0]*(pos[0,1]-pos[1,1])
		K = 12*np.abs(K)
		Area = K
		
		# Calculate gradients
		Dx_loc = np.zeros(3);Dy_loc = np.zeros(3)
		# Normalization
		#pos = (pos.T/np.sqrt(np.sum(pos**2,1))).T
		x=0;y=1
		for i in range(3):
			# Define j,k
			j=i+1;k=i+2;j=j%3;k=k%3
			
			# Calculate gradient
			Dx_loc[i] = (pos[j,y]-pos[k,y])/(2*K)
			Dy_loc[i] = (pos[k,x]-pos[j,x])/(2*K)

		# uncomment here once the local partial derivatives 
		# and the area are computed
		vals_x[e,:] = Dx_loc
		vals_y[e,:] = Dy_loc
		vals_s[e] = Area

		# saving the indices
		idx_i[e,:] = e*np.ones((3,))
		idx_j[e,:] = nodes

	Dx_coo = spsp.coo_matrix((vals_x.reshape((-1,)), 
							 (idx_i.reshape((-1,)), 
							  idx_j.reshape((-1,)))), shape=(n_t, p.shape[0]))

	Dy_coo = spsp.coo_matrix((vals_y.reshape((-1,)), 
							 (idx_i.reshape((-1,)), 
							  idx_j.reshape((-1,)))), shape=(n_t, p.shape[0]))

	surf = spsp.dia_matrix((vals_s.reshape((1,-1)), 
							np.array([0])), shape=(n_t, n_t))

	return spsp.lil_matrix(Dx_coo), spsp.lil_matrix(Dy_coo), spsp.lil_matrix(surf)  


def dtn_map(v_h, sigma_vec):

	n_bdy_pts = len(v_h.mesh.bdy_idx)
	n_pts  = v_h.mesh.p.shape[0]

	vol_idx = v_h.mesh.vol_idx
	bdy_idx = v_h.mesh.bdy_idx

	# build the stiffness matrix
	S = stiffness_matrix(v_h, sigma_vec)
	
	# reduced Stiffness matrix (only volumetric dof)
	Sb = spsp.csr_matrix(S[vol_idx,:][:,vol_idx])
	
	# the boundary data are just direct deltas at each node
	bdy_data = np.eye(n_bdy_pts)
	
	# building the rhs for the linear system
	Fb = np.zeros((n_pts - n_bdy_pts,n_bdy_pts))
	
	# solve interior dof
	U_vol = spsolve(Sb, Fb)
	
	# allocate the space for the full solution
	sol = np.zeros((n_pts,n_bdy_pts))
	
	# write the corresponding values back to the solution
	# uncomment when ready
	sol[bdy_idx,:] = bdy_data
	sol[vol_idx,:] = U_vol

	# computing the flux
	flux = S[-n_bdy_pts:,:] * sol

	# extracting the boundary data of the flux 
	DtN = flux

	# uncomment when ready
	return DtN, sol


def adjoint(v_h, sigma_vec, residual):

	n_bdy_pts = len(v_h.mesh.bdy_idx)
	n_pts  = v_h.mesh.p.shape[0]

	vol_idx = v_h.mesh.vol_idx
	bdy_idx = v_h.mesh.bdy_idx

	# build the stiffness matrix
	# given that the operator is self-adjoint
	S = stiffness_matrix(v_h, sigma_vec)
	
	# reduced Stiffness matrix (only volumetric dof)
	Sb = spsp.csr_matrix(S[vol_idx,:][:,vol_idx])
	
	# the boundary data are just direct deltas at each node
	bdy_data = np.eye(n_bdy_pts)
	bdy_data = S[-n_bdy_pts:,-n_bdy_pts:] * bdy_data
	
	# building the rhs for the linear system
	Fb = np.zeros((n_pts - n_bdy_pts,n_bdy_pts))
	
	# solve interior dof
	U_vol = spsolve(Sb, Fb)
	
	# allocate the space for the full solution
	sol_adj = np.zeros((n_pts,n_bdy_pts))
	
	# write the corresponding values back to the sol_adjution
	# uncomment when ready
	sol_adj[bdy_idx,:] = bdy_data
	sol_adj[vol_idx,:] = U_vol

	return sol_adj 


def misfit_sigma(v_h, Data, sigma_vec):
	# compute the misfit 

	# compute dtn and sol for given sigma
	dtn, sol = dtn_map(v_h, sigma_vec)

	# compute the residual
	residual  = -(Data - dtn)

	# comput the adjoint fields
	sol_adj = adjoint(v_h, sigma_vec, residual)

	# compute the derivative matrices (weakly)
	Dx, Dy, M_w = partial_deriv_matrix(v_h)

	# this should be diagonal, thus we can avoid this
	# uncomment when ready
	M_w = spsp.csr_matrix(M_w)

	Sol_adj_x = spsolve(M_w,(Dx@sol_adj))
	Sol_adj_y = spsolve(M_w,(Dy@sol_adj))

	Sol_x = spsolve(M_w,(Dx@sol))
	Sol_y = spsolve(M_w,(Dy@sol))

	# uncomment when ready
	#print(np.shape((Dx@sol_adj).T))
	grad = (Dx@sol) @ (Dx@sol_adj).T + (Dy@sol) @ (Dy@sol_adj).T

	# uncomment when ready
	return 0.5*np.sum(np.square(residual)), grad

