/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties.water;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

viscosityModel  constant;

nu               3.7e-7; // original 3.7e-7

rho              2500; // original 7578

elec_resistivity	3.25e-7; // previous = 3.25e-7 




// Thermal conductivity table [W/(m·K)]
table_kappa
(
    (300    179.8)
    (400    188.9)
    (500    198.0)
    (600    207.1)
    (800    225.3)
    (872    225.3)
    (916    90.0)    
    (1000   90.0)
    (1100   90.0)
    (1300   90.0)
    (1533   90.0)
    (1609   90.0)
    (2500   90.0)
    (5000   90.0)
);




// Specific heat capacity table [J/(kg·K)]
table_cp
(
    (300    870.8)
    (400    919.4)
    (500    968.0)
    (600    1016.6)
    (800    1113.8)
    (872    1113.8)
    (916    1170.0)  
    (1000   1170.0)
    (1100   1170.0)
    (1300   1170.0)
    (1533   1170.0)
    (1609   1170.0)
    (2500   1170.0)
    (5000   1170.0)
);




   
	Tsolidus 873;
	Tliquidus 915;
    LatentHeat 380e3;
    beta    2.32e-5; // original 2.32e-5


// ************************************************************************* //

