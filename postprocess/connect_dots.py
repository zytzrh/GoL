import openbabel
from openbabel import pybel


def connectDotsXYZ(xyz_file, outfile, connected_outfile, opt_file):
    mol = pybel.readfile("xyz", xyz_file).__next__()
    mol.write("sdf", outfile, overwrite=True)
    mol.OBMol.ConnectTheDots()
    mol.write("sdf", connected_outfile, overwrite=True)
    mol.localopt("uff")
    mol.write("sdf", opt_file, overwrite=True)
    print(mol)

def xyz2sdf(xyz_file, outfile, rawoutfile):
    mol = pybel.readfile("xyz", xyz_file).__next__()
    mol.OBMol.ConnectTheDots()
    mol.write("sdf", rawoutfile, overwrite=True)

    mol.localopt("uff")
    # mol.OBMol.ConnectTheDots() # useless

    # for i in range(3):
    #     mol.write("xyz", "tmp.xyz", overwrite=True)
    #     mol = pybel.readfile("xyz", "tmp.xyz").__next__()
    #     mol.localopt("uff")


    mol.write("sdf", outfile, overwrite=True)


if __name__ == "__main__":


    filename = "~/myutils/crossdocked_exmaple/ACRB_ECOLI_1_1048_doxorubicin_0/3aod_C_rec_3aod_rfp_lig_tt_docked_0"
    xyz2sdf(f"{filename}.xyz",
            f"{filename}_mine.sdf",
            f"{filename}_rawout.sdf")