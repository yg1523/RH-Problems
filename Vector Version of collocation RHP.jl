using LinearAlgebra
using ClassicalOrthogonalPolynomials
using SingularIntegrals
using QuadGK
using ContinuumArrays

d = 2
n = 20
P = Legendre()
x = ClassicalOrthogonalPolynomials.grid(P, n)
G = x -> [exp(-40 * x^2)  sin(x); 0 x]
N = length(x)

#calculate a list of matrix values at the collocation points.
Glist = [G(xm) - I for xm in x]

Pmat = zeros(ComplexF64, N, n)
Cmat = zeros(ComplexF64, N, n)
for m in 1:N, k in 1:n
    P_km = Legendre()[x[m], k] #evaluate the k-1 Legendre Polynomial at the m collocation point.
    Cmat[m, k] = stieltjes(P, x[m] - 0im)[k] / (-2π * im) #calculate the Cauchy Transform of the k-1 Legendre Polynomial at the m collocation point.
end

A = zeros(ComplexF64, N * d^2, n * d^2)
b = zeros(ComplexF64, N * d^2)

row_idx(j, m, r) = r + d*(m-1) + d*N*(j-1)
col_idx(j, k, q) = q + d*(k-1) + d*n*(j-1)


#every tuple (j,m,r) uniquely determines one equations
#the whole system is a tensor system, with 3 dimensions in terms of numbers of equations, and 3 dimensions in the variable u_{j,k}^{(q)}
for j in 1:d          # iterate over the rows
    for m in 1:N      # iterate over the collocation points
        Gm = Glist[m] # G(s_m) - I
        for r in 1:d  # study entry-wise on our singular integral equations.
            row = row_idx(j, m, r)
            b[row] = Gm[j, r]

            for k in 1:n
                Pkm = Pmat[m, k]
                Cpkm = Cmat[m, k]

                for q in 1:d
                    col = col_idx(j, k, q)
                    coeff = (q == r ? Pkm : 0.0) - Cpkm * Gm[q, r]
                    A[row, col] = coeff
                end
            end
        end
    end
end

x_flat = A \ b

U = zeros(ComplexF64, d, n, d)  # U[j,k,q] = u_{j,k}^{(q)}
for j in 1:d, k in 1:n, q in 1:d
    U[j, k, q] = x_flat[col_idx(j, k, q)]
end

#recover Ck from u_{j,k}^{(q)}, by stacking all u_{jk}


function build_Ck(U, k)
    d, n, _ = size(U)
    Ck = zeros(ComplexF64, d, d)
    for j in 1:d
        Ck[j, :] = U[j, k, :]  # 把 u_{j,k}^T 放入第 j 行
    end
    return Ck
end


#Recover the original function u from our matrix coefficients Ck and the orthogonal polynomial basis Pk(s).
function evaluate_u(s::ComplexF64, C_list, poly_family)
    u = zeros(ComplexF64, d, d) 
    for k in 1:n
        Pk_s = poly_family[s, k]     # the poly_family refers to a specific type of polynomials i.e. Legendre()
        u += Pk_s * C_list[k]        # each term is a d*d matrix
    end
    return u  # ∈ ℂ^{d×d}
end

P = Legendre()
C_list = [build_Ck(U, k) for k in 1:n]
s = 0.5

u_val = evaluate_u(s, C_list, P)
println(u_val)


