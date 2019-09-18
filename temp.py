shifted_boundary_regions = tuple()
region_start = wr[:2]
c1 = wr[2].stop==ss[2]-ops
c2 = wr[3].stop==ss[3]-ops
sx = ss[2]-ops-SPLITX
sy = ss[3]-ops-SPLITY
xst = wr[2].stop-wr[2].start
yst = wr[3].stop-wr[3].start
x_reg = slice(ops,xst+ops,1)
y_reg = slice(ops,yst+ops,1)
tops = 2*ops
if c1: #Bottom edge -  periodic x
    #Boundaries for up pyramid and octahedron
    shifted_boundary_regions += (region_start+(slice(xst-SPLITX,xst,1),y_reg,slice(0,SPLITX,1),wr[3]),)
if c2: #Right edge -  periodic y
    shifted_boundary_regions += (region_start+(x_reg,slice(yst-SPLITY,yst,1),wr[2],slice(0,SPLITY,1)),)
if c1 and c2:
    shifted_boundary_regions += (region_start+(slice(xst-SPLITX,xst,1),slice(yst-SPLITY,yst,1),slice(0,SPLITX,1),slice(0,SPLITY,1)),)
return shifted_boundary_regions
