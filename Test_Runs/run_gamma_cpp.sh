#INPNAME="Test_Run.k"
#INPNAME="Layered_02_Final.k"
#INPNAME="ClosedUnstructured_Final.k"
#INPNAME="LENS_025_Final.k"
INPNAME="SS316L_EBeam.k"
#INPNAME="powderBed_800k.k"
#INPNAME="YPL_Lens_0125_Final.k"
#INPNAME="YPL_3Layer_Final.k"
#INPNAME="ThinWallYPL_Final.k"
LOGFILE="logfile.log"

../build/GAMMA_cpp $INPNAME

#../build/GAMMA_cpp $INPNAME > $LOGFILE
