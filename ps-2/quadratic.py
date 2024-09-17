def usual_quadratic(a,b,c):
    delta=b**2-4*a*c
    if delta>=0 and a!=0:
        return np.array([(-b+np.sqrt(delta))/(2*a),(-b-np.sqrt(delta))/(2*a)])
    else:
        return 'Please enter diffent coefficients'

def wierd_quadratic(a,b,c):

    delta=b**2-4*a*c
    if delta>=0 and a!=0:
        return np.array([2*c/(-b-np.sqrt(delta)),2*c/(-b+np.sqrt(delta))])
    else:
        return 'Please enter diffent coefficients'
    
def quadratic(a,b,c):
     #define a treshold
    usual=usual_quadratic(a,b,c)
    weird=wierd_quadratic(a,b,c)
    if b>0:
        return np.array([usual[0],weird[1]])
    else:
        return np.array([weird[0],usual[1]])
