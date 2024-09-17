import numpy as np
import matplotlib.pyplot as plot

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

#find smallest increment for 32bit floats and check it
print('Smallest increment for 32bit:')
x_32=np.float32(1.)
y_32=np.float32(np.exp2(-23))
z_32=np.float32(np.exp2(-24))

get_fbits(x_32+y_32)

print(x_32+y_32)
print(0==z_32, x_32==x_32+z_32)

#find smallest increment for 64bit floats and check it
print('Smallest increment for 64bit:')
x_64=np.float64(1.)
y_64=np.float64(np.exp2(-52))
z_64=np.float64(np.exp2(-53))

print(x_64+y_64)
print(0==z_64, x_64==x_64+z_64)


#find smallest 32bit float and check it
print('Smallest 32bit:')
x_32=np.float32(np.exp2(-149))
z_32=np.float32(np.exp2(-150))

get_fbits(x_32)

print(x_32, z_32)

#find smallest 64bit float and check it
print('Smallest 64bit:')
x_64=np.float64(np.exp2(-1022-52))
z_64=np.float64(np.exp2(-1023-52))

print(x_64, z_64)

#find largest 32bit float and check it
print('Largest 32bit:')
x_32=np.float32(3.402823e+38)
z_32=np.float32(np.exp2(128))

get_fbits(x_32)

print(x_32, z_32)

#find largest 64bit float and check it
print('Largest 64bit:')
x_64=np.float64(1.7976931348623157e+308)
z_64=np.float64(np.exp2(1024))

print(x_64, z_64)

#alternatively, could have used the np documentation
print('Using documentation:')
print(f'32bit resolution= {np.finfo(np.float32).eps}')
print(f'32bit min= {np.finfo(np.float32).smallest_subnormal}')
print(f'32bit max= {np.finfo(np.float32).max}')
print(f'64bit resolution= {np.finfo(np.float64).eps}')
print(f'64bbit min= {np.finfo(np.float64).smallest_subnormal}')
print(f'64bit max= {np.finfo(np.float64).max}')
