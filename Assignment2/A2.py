
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

m=-4
n=3

B =np.array([1,3])
A =np.array([2,7])

I=((m*B)+(n*A))/(m+n)
E=((m*B)-(n*A))/(m+n)
#print(P)
x_AB = line_gen(A,B)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')


plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.4), B[1] * (1 - 0.5) , 'B')
plt.plot(I[0], I[1], 'o')
plt.text(I[0] * (1-0.1), I[1] * (1-0.25) , 'I')
plt.plot(E[0], E[1], 'o')
plt.text(E[0] * (1-0.05), E[1] * (1-0.05) , 'E')
print(I)
print(E)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
# plt.axis('equal')
plt.show()
