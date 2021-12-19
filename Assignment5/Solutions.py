### CBSE 10th class 2013 Question paper
# Question 7
import numpy as np
import matplotlib.pyplot as plt

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
import numpy as np
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if


# given triangle vertices

#Triangle vertices
A = np.array([1,3]) 
B = np.array([-1,0]) 
C = np.array([4,0]) 

Ar = np.linalg.det([A-B,A-C])/2
# print(Ar)

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
print("Sol 7) since area of the triangle is :", abs(Ar)," answer is option (C)")


# Question 19
import numpy as np
import matplotlib.pyplot as plt

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
import numpy as np
import matplotlib.pyplot as plt

#Triangle vertices
A = np.array([7,10]) 
B = np.array([-2,5]) 
C = np.array([3,-4]) 

#Lengths of sides 

A_B = np.linalg.norm(A-B) 
A_C = np.linalg.norm(A-C)
B_C = np.linalg.norm(C-B)

if A_B == A_C or A_B == B_C or B_C == A_C: #conditon checking if any two sides are equal
  print("Sol 19) Since the given triangle has two pairs of sides equal")
  if (np.dot(A-B,B-C)== 0 or np.dot(A-C,C-B)== 0 or np.dot(B-A,A-C)== 0): #conditon checking if any two sides are rightangle
    print("and product of a pair of sides is 0")
    print("given triangle is Isosceles and Right angled Triangle")
  else:
    print("and product of a pair of sides is not 0")
    print("given triangle is Isosceles but not Right angled Triangle")

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
