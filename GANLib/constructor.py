# -------------------
#  Constructor
# -------------------
# R - real data
# F - fake noisy data
# L - labels
# N - noise / latent representation


#AE: 
#   DE(EN(R)) -> R
 
#GAN:       
#       Mode 1:
#   D(R) -> 1       {bc}
#   D(DE.(N)) -> 0  {bc}
#   D.(DE(N)) -> 1  {bc}
#
#       Mode 2:
#   D(R), D(DE.(N)) -> 1, 0    {bc}
#   D.(DE(N)) -> 1          {bc}
#
#       Mode 3:
#   D(R), D(DE.(N)), D.(DE(N)) -> 1, 0, 1  {bc}
#
#       Mode S:
#   D(R) -> 1       {logcosh}
#   g = Gravity(D.(DE.(N)), boundaries = [-1,1], pressure = 0.5)
#   D(DE.(N)) -> g  {logcosh}
#   D(F) -> -1      {logcosh}
#   D.(DE(N)) -> 1  {logcosh}

#CGAN:
#   D(R, L) -> 1       {bc}
#   D(DE.(N), L) -> 0  {bc}
#   D.(DE(N), L) -> 1  {bc}

#DiscoGAN:  
#   D(Ra, Rb) -> 1              {mae}
#   D(DE.(Rb), EN.(Ra)) -> -1   {mae}
#   D.(DE(Rb), EN(Ra)) -> 1     {mae}
#   DE(EN(Ra)) -> Ra            {mae}
#   EN(DE(Rb)) -> Rb            {mae}

#AAE:       
#   D(N) -> 1               {bc}
#   D(EN.(R)) -> 0          (bc)
#   x = EN(R)
#   DE(x), D.(x) -> R, 1    {mse, bc}




##############