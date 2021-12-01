import ravop.core.c as R
d = R.Tensor([1,2,3])
c = R.Tensor([4,5,6])
# a = R.dot(d,c)
a = c - d
print('\nWAITING TO COMPUTE\n')
a.wait_till_computed()
print('\nGETTING STATUS\n')
print(a.get_status())
print('\nOUTPUT:\n')
print(a())