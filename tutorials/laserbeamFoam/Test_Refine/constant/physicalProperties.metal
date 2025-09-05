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
nu              5e-07;          // m2/s
rho             8000;           // kg/m3
elec_resistivity 1.0e-6;        // ohm·m

Tsolidus        1658;           // K
Tliquidus       1723;           // K
LatentHeat      2.7e5;          // J/kg
beta            5.0e-6;         // 1/K

// Thermal conductivity table [W/(m·K)]
table_kappa
(
    (300    14.9)
    (400    16.6)
    (600    19.8)
    (800    22.6)
    (1000   25.4)
    (1200   28.0)
    (1500   31.7)
    (5000   31.7)   // Clamp above last measured point
);

// Specific heat capacity table [J/(kg·K)]
table_cp
(
    (300    477.0)
    (400    515.0)
    (600    557.0)
    (800    582.0)
    (1000   611.0)
    (1200   640.0)
    (1500   682.0)
    (5000   682.0)  // Clamp above last measured point
);

// ************************************************************************* //


