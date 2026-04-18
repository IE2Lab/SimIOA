import sys

def convert_pdb(source_path, vesta_path, target_path):
    with open(source_path, 'r') as f:
        source_lines = f.readlines()

    with open(vesta_path, 'r') as f:
        vesta_lines = f.readlines()

    # Map serial number to element
    serial_to_element = {}
    for line in source_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            serial = int(line[6:11])
            name = line[12:16].strip()
            element = line[76:78].strip()
            if not element:
                element = ''.join([c for c in name if not c.isdigit()])
            serial_to_element[serial] = element.capitalize()

    output = []
    output.append("TITLE     MDANALYSIS FRAME 0: Created by Antigravity from P3FOLi_imp_f2.pdb\n")
    output.append("CRYST1   50.583   47.978   49.816  90.00  90.00  90.00 P 1           1\n")

    atom_counts = {}

    for line in source_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            serial = int(line[6:11])
            name = line[12:16].strip()
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            occ = line[54:60].strip()
            temp = line[60:66].strip()
            element = serial_to_element[serial]
            
            atom_counts[element] = atom_counts.get(element, 0) + 1
            
            if element == "Li":
                new_name = "Li"
                res_name = "Li"
                res_seq = "8"
                seg_id = "SYSTLI"
            else:
                new_name = f"{element}{atom_counts[element]}"
                res_name = "p3f"
                res_seq = "180"
                seg_id = f"SYST {element.upper():<1}"

            formatted_line = (
                f"HETATM{serial:>5} {new_name:<4} {res_name:<3} X {int(res_seq):>3}    "
                f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                f"{float(occ):6.2f}{float(temp):6.2f}      "
                f"{seg_id:<6}\n"
            )
            output.append(formatted_line)
            
    # Add CONECT records, filtering out O-H bonds
    for line in vesta_lines:
        if line.startswith("CONECT"):
            parts = line.split()
            atom1_serial = int(parts[1])
            atom1_element = serial_to_element.get(atom1_serial)
            
            others = []
            for p in parts[2:]:
                atom2_serial = int(p)
                atom2_element = serial_to_element.get(atom2_serial)
                
                # Check if it's an O-H bond
                is_oh = (atom1_element == "O" and atom2_element == "H") or \
                        (atom1_element == "H" and atom2_element == "O")
                
                if not is_oh:
                    others.append(p)
            
            if others:
                # Reconstruct CONECT line
                # PDB CONECT format: CONECT serial serial serial serial serial
                # Each serial is 5 chars wide
                new_conect = f"CONECT{atom1_serial:>5}"
                for o in others:
                    new_conect += f"{int(o):>5}"
                output.append(new_conect + "\n")
            
    output.append("END\n")
    
    with open(target_path, 'w') as f:
        f.writelines(output)

if __name__ == "__main__":
    convert_pdb(
        "/Users/riteshk/Library/CloudStorage/Box-Box/Research-postdoc/Collaboration-projects/P3XO-LMB_ED/dft/pfpeli_2.pdb",
        "/Users/riteshk/Library/CloudStorage/Box-Box/Research-postdoc/Collaboration-projects/P3XO-LMB_ED/dft/pfpeli_2_.pdb",
        "/Users/riteshk/Library/CloudStorage/Box-Box/Research-postdoc/Collaboration-projects/P3XO-LMB_ED/dft/pfpeli_2_converted.pdb"
    )