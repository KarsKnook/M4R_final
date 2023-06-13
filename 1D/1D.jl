using LinearAlgebra, BandedMatrices, BlockArrays
import LinearAlgebra.cholesky


"Data type for matrix arising in Poisson problem"
struct PoissonMatrix{T} <: AbstractMatrix{T}
    Hat1::Symmetric{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}  # top left tridiagonal block
    B::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}  # diagonal block
end


function Base.size(A::PoissonMatrix)
    n = size(A.Hat1, 1) - 1  # amount of cells
    p = size(A.B, 1) - 1  # highest W_p

    dim = (n + 1) + n*(p + 1)
    (dim, dim)
end


function Base.getindex(A::PoissonMatrix, i::Int, j::Int)
    n = size(A.Hat1, 1) - 1
    p = size(A.B, 1) - 1
    cutoff = n + 1  # index of last hat function

    if i ≤ cutoff && j ≤ cutoff  # indexing of Hat1
        A.Hat1[i, j]
    elseif i == j  # indexing of B
        A.B[div(i-cutoff-1, n)+1, div(i-cutoff-1, n)+1]
    else
        0
    end
end


"Left division for solving Ax = b"
function Base.:\(A::PoissonMatrix, b::BlockArrays.PseudoBlockVector)
    n = size(A.Hat1, 1) - 1
    p = size(A.B, 1) - 1

    diag_solves = [A.B[i, i]\b[Block(i+1)] for i in 1:p+1]  # solve the diagonal block
    [cholesky(A.Hat1)\b[Block(1)]; [(diag_solves...)...]]  # concatenate the tridiagonal solve with diagonal solve
end


"Data type for matrix arising in Helmholtz problem"
struct HelmholtzMatrix{T} <: AbstractMatrix{T}
    Hat1::Symmetric{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}  # bottom right block
    Hat2::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}  # [end-1, end] block
    Hat3::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}  # [end-2, end] block
    B::Symmetric{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}  # diagonal block
end


function Base.size(A::HelmholtzMatrix)
    n = size(A.Hat1, 1) - 1
    p = size(A.B, 1) - 1

    dim = (n + 1) + n*(p + 1) 
    (dim, dim)
end


function Base.getindex(A::HelmholtzMatrix, i::Int, j::Int)
    n = size(A.Hat1, 1) - 1
    p = size(A.B, 1) - 1
    cutoff = n*(p + 1)  # index of last bubble function

    if i > j  # by symmetry
        A[j, i]
    elseif j > cutoff # indexing of Hat
        if i > cutoff  # indexing of Hat1
            A.Hat1[mod(i, cutoff), mod(j, cutoff)]
        elseif i > cutoff - n  # indexing of Hat2
            A.Hat2[mod(i, cutoff-n), mod(j, cutoff)]
        elseif i > cutoff - 2n  # indexing of Hat3
            A.Hat3[mod(i, cutoff-2n), mod(j, cutoff)]
        else
            0
        end
    elseif i == j  # indexing of diagonal of B
        A.B[div(i-1, n)+1, div(i-1, n)+1]
    elseif i + 2*n == j  # indexing of upper diagonal of B
        A.B[div(i-1, n)+1, div(i-1, n)+3]
    else
        0
    end
end


function Base.:*(a::Number, A::HelmholtzMatrix)
    HelmholtzMatrix(a*A.Hat1, a*A.Hat2, a*A.Hat3, a*A.B)
end


function Base.:+(A::HelmholtzMatrix, B::HelmholtzMatrix)
    HelmholtzMatrix(A.Hat1 + B.Hat1, A.Hat2 + B.Hat2, A.Hat3 + B.Hat3, A.B + B.B)
end


function Base.:-(A::HelmholtzMatrix, B::HelmholtzMatrix)
    A + (-1)*B
end


"Addition of a Helmholtz and Poisson matrix"
function Base.:+(A::HelmholtzMatrix, B::PoissonMatrix) # basis functions are ordered reversely for Poisson matrix
    Hat1 = A.Hat1 + B.Hat1  # B.Hat1 is persymmetric so no reversion needed
    B = A.B + Diagonal([B.B[i,i] for i = 1:size(B.B)[1]][end:-1:1])  # reversing B.B
    HelmholtzMatrix(Hat1, A.Hat2, A.Hat3, B)  # output will be a Helmholtz matrix
end


function Base.:-(A::PoissonMatrix, B::HelmholtzMatrix)
    (-1)*B + A
end


"Data type to store upper factor of Cholesky decomposition of a HelmholtzMatrix"
struct HelmholtzMatrixCholesky{T} <: AbstractMatrix{T}
    chol_Hat1::UpperTriangular{T, Matrix{T}}
    chol_Hat2::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}
    chol_Hat3::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}
    chol_B::UpperTriangular{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}
end


function Base.size(A::HelmholtzMatrixCholesky)
    n = size(A.chol_Hat1, 1) - 1
    p = size(A.chol_B, 1) - 1

    dim = (n + 1) + n*(p + 1) 
    (dim, dim)
end


"Applies L_B^{-1} to C^T in block Cholesky algorithm"
function Base.:\(chol_B::Cholesky, C::Tuple)
    n = size(C[1], 1)
    p = size(chol_B, 1) - 1

    # equivalent to doing diagonal solves on the last two blocks
    (chol_B.L[p+1, p+1] \ C[1], chol_B.L[p, p] \ C[2])
end


"Computes Cholesky decomposition of a HelmholtzMatrix"
function cholesky(A::HelmholtzMatrix)
    chol_B  = cholesky(A.B)  # cholesky of B

    (chol_Hat2, chol_Hat3) = chol_B \ (A.Hat2, A.Hat3)  # computing the Cholesky blocks corresponding to Hat2 and Hat3
    S = A.Hat1 - [chol_Hat2; chol_Hat3]'*[chol_Hat2; chol_Hat3]  # computing the Schur complement
    chol_Hat1 = cholesky(S)  # computing Cholesky of Schur complement

    HelmholtzMatrixCholesky(chol_Hat1.U, chol_Hat2, chol_Hat3, chol_B.U)
end


"Solves chol_A*x = b where chol_A is the Cholesky decomposiition of A"
function Base.:\(chol_A::HelmholtzMatrixCholesky, b::AbstractVector)
    n = size(chol_A.chol_Hat1, 1) - 1
    p = size(chol_A.chol_B, 1) - 1

    # forward sub
    y1 = [chol_A.chol_B' \ b[i:n:n*p+i] for i = 1:n]  # solve upper left block per cell
    y1 = collect(Iterators.flatten(zip(y1...)))  # combine the solves ordered per polynomial degree
    rhs = b[end-n:end] - (chol_A.chol_Hat3'*y1[end-2n+1:end-n] + chol_A.chol_Hat2'*y1[end-n+1:end])  # compute rhs in 2x2 elimination, L_21 is sparse
    y2 = chol_A.chol_Hat1' \ rhs  # solve for the lower half of y

    # backward sub
    x2 = chol_A.chol_Hat1 \ y2  # solve for
    y1[end-2n+1:end-n] -= chol_A.chol_Hat3*x2  # update y1 which will be used as rhs in the elimination, L21 is sparse
    y1[end-n+1:end] -= chol_A.chol_Hat2*x2  # update y1 which will be used as rhs in 2x2 elimination, L21 is sparse
    x1 = [chol_A.chol_B \ y1[i:n:n*p+i] for i = 1:n]  # solve per cell
    x1 = collect(Iterators.flatten(zip(x1...)))  # combine the solves ordered per polynomial degree

    [x1; x2]
end


"Solves chol_A*X = B where chol_A is the Cholesky decomposiition of A"
function Base.:\(chol_A::HelmholtzMatrixCholesky, B::Matrix)
    reduce(hcat, [chol_A \ B[1:end, i] for i=1:size(B)[1]])
end


"Solves X*chol_A = B where chol_A is the Cholesky decomposiition of A"
function Base.:/(B::Matrix, chol_A::HelmholtzMatrixCholesky)
    B = Matrix(B')
    (chol_A \ B)'  # HelmholtzMatrix is symmetric
end


"Constructing Poisson matrix"
function construct_Poisson_matrix(n, p, D₂, bcs)
    B = BandedMatrix(D₂[n+2:n:n*(p+1)+2, n+2:n:n*(p+1)+2], (0, 0))  # up to W_p
    Hat1 = D₂[Block(1), Block(1)]  # hat functions

    if bcs == "dirichlet"
        Hat1[:,1] .= 0; Hat1[1,:] .= 0; Hat1[1,1] = 1; Hat1[:,n+1] .= 0; Hat1[n+1,:] .= 0; Hat1[n+1,n+1] = 1
    end
    
    PoissonMatrix(Symmetric(Hat1), B)
end


"Constructing Helmholtz matrix"
function construct_Helmholtz_matrix(n, p, A, bcs)
    diag = [A[Block(i), Block(i)][1,1] for i in 2:p+2][end:-1:1]  # diagonal of W_k up to degree p and reversing
    off_diag = [A[Block(i), Block(i+2)][1,1] for i in 2:p][end:-1:1] # off-diagonal of W_k up to degree p and reversing
    B = Symmetric(BandedMatrix((-2=>off_diag, 0=>diag, 2=>off_diag), (p+1, p+1)))  # create B

    Hat1 = A[Block(1), Block(1)]
    Hat2 = A[Block(2), Block(1)]
    Hat3 = A[Block(3), Block(1)]

    if bcs == "dirichlet"
        Hat1[1,:] .= 0; Hat1[n+1,:] .= 0; Hat1[:, 1] .= 0; Hat1[:, n+1] .= 0; Hat1[1,1] = 1; Hat1[n+1, n+1] = 1;
        Hat2[:, 1] .= 0; Hat2[:, n+1] .= 0;
        Hat3[:, 1] .= 0; Hat3[:, n+1] .= 0;
    end

    Hat1 = Symmetric(BandedMatrix(Hat1[end:-1:1, end:-1:1], (1,1)))
    Hat2 = BandedMatrix(Hat2[end:-1:1, end:-1:1], (0,1))
    Hat3 = BandedMatrix(Hat3[end:-1:1, end:-1:1], (0,1))
    
    HelmholtzMatrix(Hat1, Hat2, Hat3, B)
end


"Constructing RHS vector b"
function construct_b(n, p, C, M, f, bcs)
    f = C \ f  # interpolate f into C
    b = M*f
    b = b[Block.(1:p+2)]  # up to W_p

    if bcs == "dirichlet"
        b[1] = 0; b[n+1] = 0
    end

    b
end


function L2_norm_1D(f, M)
    sqrt(f'*M*f)
end