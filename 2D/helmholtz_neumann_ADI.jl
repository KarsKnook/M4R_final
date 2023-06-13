using PiecewiseOrthogonalPolynomials, Plots, LinearAlgebra, ClassicalOrthogonalPolynomials, StaticArrays, Arpack, SparseArrays, LaTeXStrings
import PiecewiseOrthogonalPolynomials: ∞

"""
Solving k²<u,v> + <∇u, ∇v> = <f,v> on [-1,1]^2 using ADI method
"""

include("2D.jl")
include("../1D/1D.jl")

k = 4
n = 4  # amount of cells in each direction
dx = 2/n  # width of cell
p = 8  # degree, up to W_p
N = n+1 + n*(p+1)  # amount of basis functions in C
tol = 1e-8

r = range(-1, 1; length=n+1)
C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2) * P_k^(1,1)
P = ContinuousPolynomial{0}(r)  # piecewise Legendre
CtP = C'*P
PtP = P'*P
PinvC = P\C
x = axes(C, 1)
y = axes(C, 1)
D = Derivative(x)
M = C'*C  # mass matrix
D₂ = (D*C)'*(D*C)  # weak Laplacian

M¹ = construct_Helmholtz_matrix(n, p, M, "neumann")
D₂¹ = construct_Poisson_matrix(n, p, D₂, "neumann")

# finding spectral bounds
c = k^2/2
d = k^2/2 + D₂¹[end, end]/Real(Arpack.eigs(sparse(M¹), nev = 1, which=:SM)[1][1])
a = -d  # spectral ranges are opposite
b = -c

# finding RHS
u_exact = z -> ((x,y)= z; cos.(π*x)*cos.(π*y)); f = z -> ((x,y)= z; (k^2 + 2π^2)*cos.(π*x)*cos.(π*y))
F = interpolate_2D(f, n, p+2)  # interpolate F into P⊗P
F = CtP[1:N, 1:n*(p+2)]*F*(CtP[1:N, 1:n*(p+2)])'  # RHS <f,v>
F = F[end:-1:1, end:-1:1]  # reverse F because basis functions are ordered reversely

U = ADI_generalised(k^2/2*M¹ + D₂¹, M¹, -1*M¹, k^2/2*M¹ + D₂¹, F, a, b, c, d, tol)  # solve
U = U[end:-1:1, end:-1:1]  # undo reversion
u = (x,y) -> C[x, 1:N]'*U*C[y, 1:N]  # construct solution

# computing L2 norm
U_legendre = PinvC[Block.(1:p+2), Block.(1:p+2)]*U*PinvC[Block.(1:p+2), Block.(1:p+2)]' # transform coefficients of numerical solution from C⊗C to P⊗P
U_exact = interpolate_2D(u_exact, n, 2*p+2) # double the interpolation degree of exact solution for accuracy
U_exact[1:size(U_legendre)[1], 1:size(U_legendre)[2]] -= U_legendre  # compute difference
L2 = L2_norm_2D(U_exact, (P'P)[Block.(1:2p+2), Block.(1:2p+2)]) # compute L2 norm of difference
L2 = round(L2, sigdigits=3) # round to 3 significant digits

# plot
x = y = range(-1,1; length=50)
contourf(x, y, u.(x', y), xlabel=L"x", ylabel=L"y", title="L^2 = $L2")