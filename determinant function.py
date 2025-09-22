# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:36:33 2025

@author: lucas
"""

import math 
import numpy as np
import sympy as sp


class matrix: 
    def __init__(self): 
        self.rows = int(input("Enter the number of rows: "))
        self.cols = int(input("Enter the number of columns: "))
        self.data = []
        
        print(f"Enter the matrix elements ({self.rows} x {self.cols}): ")
        for i in range(self.rows): 
            row = []
            for j in range(self.cols): 
                element = int(input(f"Enter the element(element [{i}][{j}]: "))
                row.append(element)
            self.data.append(row)
        
    def display(self): 
        ##displays the matrix 
        print("Matrix: ")
        for row in self.data:
            print([f"{x:8.2f}" for x in row])
    
    def get_data(self): 
        return self.data
    
    def get_dimensions(self):
        return self.rows, self.cols
    
    def __getitem__(self, idx):
       return self.data[idx]

def zero_vector(A): 
    return data_matrix([0] for x in A)
    
def colvec(v):                          # list -> n×1 column (as your matrix type)
    return data_matrix([[x] for x in v])

def flatten_col(M):                     # m×1 matrix -> flat list length m
    return [row[0] for row in M.get_data()]

def scalar(num, A): 
    n, m = A.get_dimensions()
    result = []
    s = 0
    for i in range(n): 
        row = []
        for j in  range(m): 
             row.append(num * A[i][j])
        result.append(row)
    return data_matrix(result)

def matrix_multiply(A, B):
   #matrix multiplication function that takes two matrices as an argument and dots its columns w/rows
    n1, m1 = A.get_dimensions()
    n2, m2 = B.get_dimensions()
    result = []
    
    if m1 != n2:
        raise ValueError("Columns of matrix (A) must equal rows of matrix (B)")
    else: 
        for i in range(n1):
            row = []
            for j in range(m2): 
                s = 0
                for k in range(m1):
                    s += A[i][k] * B[k][j]
                row.append(s)
            result.append(row)
    return data_matrix(result) 


def matrix_subtract(A, B): 
    n1, m1 = A.get_dimensions()
    n2, m2 = B.get_dimensions()
    result = []
    
    if n1 != n2 and m1 != m2: 
        raise ValueError("Matrices must have the same dimensions!")
    else: 
        for i in range(n1):
            row = []
            for j in range(n2): 
                row.append(A[i][j] - B[i][j])
            result.append(row)
    return result 
        
def matrix_add(A, B): 
    n1, m1 = A.get_dimensions()
    n2, m2 = B.get_dimensions()
    result = []
    
    if n1 != n2 and m1 != m2: 
        raise ValueError("Matrices must have the same dimensions!")
    else: 
        for i in range(n1):
            row = []
            for j in range(m1): 
                row.append(A[i][j] + B[i][j])
            result.append(row)
    return result 
   
def determinant_2x2(matrix): 
    a,b = matrix.get_data()[0]
    c,d = matrix.get_data()[1]
    return a*d - b*c

def identity_matrix(n):
    """Create an n×n identity matrix as your TempMatrix type."""
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1 if i == j else 0)
        data.append(row)
    return data_matrix(data)  # reuse your helper to wrap it

def snap(v, tol=1e-12):
    out = []
    for x in v:
        # 1) Convert SymPy types to numeric
        if hasattr(x, "as_real_imag"):               # SymPy number/expr
            a, b = x.as_real_imag()
            ar, br = float(a), float(b)
            val = complex(ar, br)
        else:
            # Python numeric: make it complex uniformly
            val = complex(x) if not isinstance(x, complex) else x

        # 2) Zero-out tiny parts
        re = 0.0 if abs(val.real) < tol else val.real
        im = 0.0 if abs(val.imag) < tol else val.imag

        # 3) If imag vanished, return float; else return complex
        out.append(re if im == 0.0 else complex(re, im))
    return out

def determinant(matrix):
    data = matrix.get_data()
    rows, cols = matrix.get_dimensions()

    if rows == 1: 
        return data[0][0]
    elif rows == 2:  # FIXED: Added 2x2 base case
        return (data[0][0] * data[1][1]) - (data[0][1] * data[1][0])
    else: 
        i = 0
        det = 0
        for j in range(cols):
            new_matrix = []
            for row_index in range(rows):
                if row_index != i: 
                    new_row = []
                    for col_index in range(cols):
                        if col_index != j:
                            new_row.append(data[row_index][col_index])
                    new_matrix.append(new_row)
            sign = (-1)**(i + j)
            print("sign: ",sign)
        
            element = data[i][j]
            print("elemnt: ", element)
        
            temp_matrix = data_matrix(new_matrix)
                
            print("temp_matrix: ", temp_matrix)
            sub_determinant = determinant(temp_matrix)
                
            print("sub_determinant: ", sub_determinant)
            
            det += sign * element * sub_determinant
                 
    return det

def det_LU(matrix):
    #logs every step that applies to identity matrix for LU determinant with O(n^3)
    #uses gaussian elimination from eigenvalues solver function
    log = []
    M = [row[:] for row in matrix.get_data()]
    n, m = matrix.get_dimensions()
    r = 0 
    tol = 1e-9
    swap = 0
    
        #current pivot row for the current matrix
    for c in range(m):
        pivot_row = None
        best = 0.0
        for rr in range(r, n):
            val = M[rr][c]
            if abs(val) > tol and abs(val) > best: 
                best = abs(val)
                pivot_row = rr 
        if pivot_row is None or best <= tol:
            return 0.0, log
            #swaps pivots if there is a 0 below the row
        if pivot_row != r: 
            temp_row = M[r]
            M[r] = M[pivot_row]
            M[pivot_row] = temp_row
            log.append(("swap", r, pivot_row))
            swap ^= 1
            
        pivot_entry = M[r][c]
        if abs(pivot_entry) <= tol:
            r += 1
            if r == n: break
            continue
        #eliminates the next row
        for rr in range(r+1, n): 
            factor = M[rr][c] / pivot_entry
            if rr == r: 
                continue
            for j in range(c, n):          # update from c onward
                M[rr][j] -= factor * M[r][j]
            log.append(("add", rr, r, -factor))
            
        r += 1
        if r == n:
            break
        print("Log: ", log)
         
    detU = 1.0
    for i in range(n):
        detU *= M[i][i]
    print(detU)
    detA = -detU if (swap & 1) else detU
    print("Det A: ", detA)
    return float(detA), log
        
def data_matrix(data):
    """Helper function to create a Matrix object from 2D list data"""
    class TempMatrix:
        def __init__(self, matrix_data):
            self.data = matrix_data
            self.rows = len(matrix_data)
            if matrix_data and len(matrix_data) > 0 and isinstance(matrix_data[0], list):
                self.cols = len(matrix_data[0])
            else:
                self.cols = 0
        
        def get_data(self):
            return self.data
        
        def get_dimensions(self):
            return self.rows, self.cols
        
        def __getitem__(self, idx):
            return self.data[idx] 
    """
        def __str__(self):
            if not hasattr(self, 'data') or not self.data:
                return "[]"
            lines = []
            for row in self.data:
                # Format each number nicely
                lines.append(" ".join(f"{x:8.2f}" if isinstance(x, (int, float)) else str(x) for x in row))
            return "\n".join(lines)
    """
    return TempMatrix(data)
    
def transpose(M): 
    n, m = M.get_dimensions()
    M_T = [[M[j][i] for j in range(n)] for i in range(m)]
    return data_matrix(M_T)

def inverse(A): 
    M = A.get_data()
    n, m = A.get_dimensions()
    I_M = [[1 if i == j else 0.0 for j in range(n)] for i in range(n)]
    #tolerance to snap noise
    tol = 1e-9
    #RREF algorithm that makes the first row the pivot point at the beginning
    r = 0 
    for c in range(m):
        pivot_row = None
        best = 0.0
        for rr in range(r, n):
            val = M[rr][c]
            if abs(val) > tol and abs(val) > best: 
                best = abs(val)
                pivot_row = rr 
        if pivot_row is None:
            continue
            #swaps pivots if there is a 0 below the row
        if pivot_row != r: 
            M[r], M[pivot_row] = M[pivot_row], M[r]
            I_M[r], I_M[pivot_row] = I_M[pivot_row], I_M[r]

        #normalizes rows
        pivot_entry = M[r][c]
        if abs(pivot_entry) <= tol:
            continue  # defensive: degenerate pivot
        #only operates from c onward (faster and safer)
        s = 1.0 / pivot_entry
        M[r] = [x * s for x in M[r]]
        #deletes noise 
        M[r] = [0 if abs(x) < tol else x for x in M[r]]
        I_M[r] = [x * s for x in I_M[r]]
        I_M[r] = [0 if abs(x) < tol else x for x in I_M[r]]
        #eliminates the next row
        for rr in range(n): 
            factor = M[rr][c]
            if rr == r: 
                continue
            if abs(factor) > tol:
                new_row = [a - factor * b for a, b in zip(M[rr], M[r])]
                new_row = [0 if abs(x) < tol else x for x in new_row]
                M[rr] = new_row
                I_M[rr] = [a - factor * b for a, b in zip (I_M[rr], I_M[r])]
                I_M[rr] = [0 if abs(x) < tol else x for x in I_M[rr]]
        r += 1
        if r == n: 
            break 
    print("(now identity) original matrix: ",  M)
    print("Inverted matrix: ", I_M)
    
    return data_matrix(I_M)

        
#this solves for the nullspace of any rref matrix (must be rref already!!)
def solve_kernel(matrix, tol = 1e-9):
    n, m = matrix.get_dimensions()
    M = matrix.get_data()
    #empty pivots list
    pivots = []
    #finds pivot columns
    for i in range(n): 
        p = None
        for j in range(m):
            if abs(M[i][j]) > tol: 
                p = j
                break 
        if p is not None: 
            pivots.append(p)
  #  print("Pivots:", pivots)
    #identifies free columns (any colmn thats not in pivots)    
    free_cols = [j for j in range(m) if j not in pivots]
  #  print("Free cols:", free_cols)
    
    basis = []
    for i in free_cols: 
        #x is the 0 vector with length of number of columns
        x = [0.0] * m
        #it holds an arbitrary value of 1
        x[i] = 1
     #   print("X: ", x)
        #finds the pivot columns, and backsolves
        #sets everything equal to x(pivot) since it is always 1, convenient 
        #then defines this as the basis vectors
        for r, pc, in enumerate(pivots):
            s = 0.0 
            for c in free_cols: 
                coeff = M[r][c]
                if abs(coeff) > tol: 
                    s += coeff * x[c]
            x[pc] = -s
        basis.append(x)
        
    def snap(v, tol=1e-12):
        return [0.0 if abs(x) < tol else float(x) for x in v]
    
        
 #   print("basis: ",  basis)

    return basis
                    
def calculate_eigenvectors(matrix): 
    A = matrix.get_data()
    n, m = matrix.get_dimensions()
    lam = sp.symbols('λ')
   # print(lam)
    if n != m: 
        raise ValueError("Determinant only works for square matrices!")
    amli = []
    #this first portion handles A-lam(I)
    for i in range(n):
        row = []
        for j in range(m): 
            value = sp.Integer(A[i][j])
            if i == j: 
                row.append(value - lam)
                #col.append(value - lam)
            else: 
                row.append(value)
                #col.append(value)
        amli.append(row)
    
    #i print the a-lambdaI
  #  print(amli)
    #uses sympy to solve characteristic polynomial, however will later edit this to solve polynomial from scratch
    amli_obj = data_matrix(amli)
    poly = determinant(amli_obj)
    poly = sp.expand(poly)
   # print("Characteristic polynomial: ", poly)
    eigenvalues_dict = sp.roots(poly)
   # print("Eigenvalues dictionary: ", eigenvalues_dict)
    eigenvalues = [ev for ev, mult in eigenvalues_dict.items() for _ in range(mult)]
   # print("Eigenvalues list: ", eigenvalues)
    
    def to_py_number(ev, prec=12, tol=1e-12):
        evN = sp.N(ev, prec)
        a, b = evN.as_real_imag()
        ar, br = float(a), float(b)
        if abs(br) < tol:
            return ar                 # return plain float if imag part is tiny
        else:
            return complex(ar, br)    # return Python complex if imag is significant

    eigs = [to_py_number(ev) for ev in eigenvalues]
   # print("Python eigenvalues: ", eigs)
    #now that i have my eigenvalues, i plug them back into my new matrix that will be set equal to 0 
    
    amli_new = []
    for i in range(len(eigs)): 
        M = []
        for j in range(n):
            row = []
            for k in range(m):
                value = sp.Integer(A[j][k])
                if j == k: 
                    row.append(value - eigs[i])
                else: 
                    row.append(value)
            M.append(row)
        amli_new.append(M)
   # print("Amli new: ", amli_new)
    #print("M:", M)
    
    #to make sure that the tolerance of "TO PY NUMBER" is small enough
    evs_complex = [complex(sp.N(ev, 12)) for ev in eigenvalues]
    #print(evs_complex)
    
    #tolerance for real numbers in RREF 
    tol = 1e-9
    #RREF algorithm that makes the first row the pivot point at the beginning
    for i in range(len(amli_new)): 
        M = amli_new[i]
        nrows = len(M)
        ncols = len(M[0])
        #current pivot row for the current matrix
        r = 0 
        for c in range(ncols):
            pivot_row = None
            best = 0.0
            for rr in range(r, nrows):
                val = M[rr][c]
                if abs(val) > tol and abs(val) > best: 
                    best = abs(val)
                    pivot_row = rr 
            if pivot_row is None:
                continue
                #swaps pivots if there is a 0 below the row
            if pivot_row != r: 
                temp_row = M[r]
                M[r] = M[pivot_row]
                M[pivot_row] = temp_row     
            #normalizes rows
            pivot_entry = M[r][c]
            if abs(pivot_entry) <= tol:
                continue  # defensive: degenerate pivot
            #only operates from c onward (faster and safer)
            M[r][c:] = [x / pivot_entry for x in M[r][c:]]
            # optional: snap tiny noise
            M[r][c:] = [0 if abs(x) < tol else x for x in M[r][c:]]

            #eliminates the next row
            for rr in range(nrows): 
                factor = M[rr][c]
                if rr == r: 
                    continue
                if abs(factor) > tol:
                    new_row = [a - factor * b for a, b in zip(M[rr], M[r])]
                    new_row = [0 if abs(x) < tol else x for x in new_row]
                    M[rr] = new_row
            r += 1
            if r == nrows:
                break
   # print("Rref Matrix: ", amli_new)
    
    #calculates nullspace:
    
    eigenspaces = []
    for idx, M in enumerate(amli_new):
        basis = solve_kernel(data_matrix(M), tol=1e-9)
        basis = [snap(v) for v in basis]
        eigenspaces.append((eigs[idx], basis))

        V_cols = []
    for lam, basis in eigenspaces:
        if not basis:
            # numerical edge case: fall back to standard basis or zeros
            raise ValueError("No eigenvector found for eigenvalue {}".format(lam))
        v = basis[0]                 # pick first basis vector for this λ
        # normalize
        norm = math.sqrt(sum(float(x)*float(x) for x in v))
        v = [float(x)/norm if norm > 0 else float(x) for x in v]
        V_cols.append(v)

    return eigs, V_cols


def SVD(A): 
    tol = 1e-9
    n, m = A.get_dimensions()
    sigma = 0
    sigma_list = []
    A_T = transpose(A)
    ATA = matrix_multiply(A_T, A)
    eigenvals, V_cols, = calculate_eigenvectors(ATA)
    
    #loop that goes through each eigenvalue and calculates sigma
    for i in eigenvals:
        val = i.real if hasattr(i, "real") else float(i)
        val = max(0.0, val) 
        sigma = math.sqrt(val)
        sigma_list.append(sigma)
    print("Singular value(s): ", sigma_list)
    
    order = sorted(range(len(sigma_list)), key=lambda i: sigma_list[i], reverse=True)
    sigma_list = [sigma_list[i] for i in order]
    V_cols = [V_cols[i]     for i in order]
    
    #first, sigma matrix
    sigma_matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == j and j < len(sigma_list): 
                row.append(sigma_list[i])
            else:
                row.append(0.0)
                
        sigma_matrix.append(row)
        
    #wrap it in matrix class  
    Σ = data_matrix(sigma_matrix)
    print("Sigma: ", Σ.get_data())
    
    #creating U matrix 
    U_cols = []
    for sigma, v in zip(sigma_list, V_cols): 
        if sigma > tol: 
            Av = matrix_multiply(A, colvec(v)).get_data()
            u = [row[0] / sigma for row in Av]
            unorm = math.sqrt(sum(x*x for x in u))
        else:
            u = [0.0]*n
        U_cols.append(u)
    #wrapping list of "U_COLS" into matrix class
    U = data_matrix(U_cols)
    
    #now need to make V matrix 
    #v matrix is composed of eigenvectors of ATA as columns
    #since we already found V earlier in the function by calling calculate eigenvectors,
    #we just wrap it in the matrix class (and then find transpose)

    V = data_matrix([list(col) for col in zip(*V_cols)])
    V_T = transpose(V)
    
    print("V: ", V.get_data())
    print("V transposed: ", V_T.get_data())

    #now to check if A = U*sigma*VT
    x = matrix_multiply(U, Σ)
    A_recreated = matrix_multiply(x, V_T)
    print("Checking if A = UsigmaV_T: ", A_recreated.get_data())

    return Σ, V, V_T, U, A_recreated 


#main function   
def main(): 
    print("=== Matrix Calculator ===")
    
    # Create a matrix with user input
    m = matrix()
    # Display the matrix
    m.display()
    
    scaled = scalar(3, m)
    print("Scaled matrix: ", scaled.get_data())
    inv = inverse(scaled)
    print("Inverse matrix: ", inv.get_data())
    AA_1 = matrix_multiply(m, inv)
    print("AA_1: ", AA_1.get_data())

    
    
    """
    choice = input("Compute inverse first? (y/n): ").strip().lower()
    if choice == 'y':
        inv = inverse(m)
        if inv is not None:
            print("Inverse:", inv)
            # exit early to skip rest
            return
        else:
            print("Matrix is singular; continuing...")
    """   
    # Calculate determinant if square matrix

    """
    rows, cols = m.get_dimensions()
    if rows == cols:
        det = determinant(m)
        eigenpairs, eigenvalues = calculate_eigenvectors(m)
        determinant_LU = det_LU(m)
        print(f"Determinant: {det}")
        print(f"{eigenvalues}")
        print(f"Determinant_LU: {determinant_LU}")
        print("Eigenpairs (λ, basis):")
        for lam, basis in eigenpairs:
                print("λ ≈", lam)
                for v in basis:
                    print("  v ~", v)
    else:
        print("Matrix is not square - determinant not available")

    """
    #calculating svd and checking whether it works 

    SVD(m)

main()