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
// *                  ** //

viscosityModel  constant;
nu              5e-07;
rho             8000;
Tsolidus        1658;
Tliquidus       1723;
LatentHeat      2.7e5;
beta            5.0e-6;

//    poly_kappa   (25 0.0 0 0 0 0 0 0);
//    poly_cp   (700 0.0 0 0 0 0 0 0);

elec_resistivity    1.0e-6;

table_kappa
(
    (273.15  401.0)
    (300.0   398.0)
    (400.0   393.0)
    (500.0   386.0)
    (600.0   379.0)
    (700.0   371.0)
    (800.0   364.0)
    (900.0   356.0)
    (1000.0  349.0)
    (5000.0  349.0)
);

table_cp
(
    (273.15  385.0)
    (300.0   387.0)
    (400.0   393.0)
    (500.0   417.0)
    (600.0   442.0)
    (700.0   468.0)
    (800.0   494.0)
    (900.0   520.0)
    (1000.0  545.0)
    (5000.0  349.0)
);

// Piecewise kappa(T): Temperature [K], Thermal conductivity [W/(m·K)]
// ************************************************************************* //
