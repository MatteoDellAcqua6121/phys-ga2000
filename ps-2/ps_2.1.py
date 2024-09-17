import numpy as np
import matplotlib.pyplot as plot

#100.98763

#define a function that convert from a list of bit to a base ten number
def base2(x):
    x=np.flip(x)
    j=0
    k=0
    for i in x:
        j+=i*2**(k)
        k+=1

    return(j)

#Imports the get_bits and get_fbits (for floats) from the Jupiter notebook of the course
def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

def get_fbits(number):
    """For a NumPy float, return bit representation devided into sign, mantissa and exponent
    
    Inputs:
    ------
    number : NumPy float32
        value to convert into list of bits
        
    Returns:
    -------
    prints in readable english the bits devided into sign, exponent and mantissa
    """
    bitlist=get_bits(np.float32(number))
    sign = bitlist[0]
    exponent = bitlist[1:9]
    r_exponent=base2(exponent)
    mantissa = bitlist[9:32]
    r_mantissa=base2(mantissa)
    template = """{value} decimal ->
        sign = {sign} 
        exponent = {exponent} = {r_exponent}
        mantissa = {mantissa} = {r_mantissa}"""
    print(template.format(value=number, sign=sign, exponent=exponent, mantissa=mantissa,r_exponent=r_exponent,r_mantissa=r_mantissa ))

get_fbits(100.98763)

#sign = 0
#exponent = [1, 0, 0, 0, 0, 1, 0, 1]=133
#mantissa = [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]=4848043

#print(np.float32(1+4848043*2**(-23))*2**(133-127))
#this is equal to 100.98763275146484

#print((2**23+4848043)*2**6)
#or as a fraction 847145664/8388608

print(f"The number which is actually stored = {np.float32((1+4848043*2**(-23))*2**(133-127))} or 847145664/8388608")

print('Difference =', 100.98763275146484-100.98763)
#thus the difference is 2.7514648479609605e-06

