"""
LBLRTM TAPE5 Writer (Exact MATLAB to Python Translation)
Author: Paolo Antonelli (paolo.antonelli@adaptivemeteo.com)
Copyright: AdaptiveMeteo S.r.l.
Version: 1.0
Date: 2025-05-01

Dependencies:
- numpy

Description:
This function creates a TAPE5 file for the LBLRTM radiative transfer model from a given atmospheric profile.

Parameters:
- prof: dictionary with fields:
    - pressure: ndarray(N), in hPa
    - tdry: ndarray(N), temperature in K
    - w: ndarray(N), water vapor in g/kg
    - alt: ndarray(N), altitude in km
    - oz: ndarray(N), ozone in ppmv
- comment: string, comment line for the TAPE5
- VBOUND: list of 2 floats, beginning and ending wavenumber
- TBOUND: list of 2 floats, [surface temperature, emissivity] (emissivity ignored here)
- HBOUND: list of 3 floats, [observer_alt, end_alt, zenith_angle]
- aflag: int, standard atmosphere code (1–6)
- CO2amt: float or ndarray(N), CO2 concentration (ppmv)
- gcflag: int, gas combination flag (1–6)
- selwn: unused here, but normally refers to selected wavenumber range
- outfile_type: int, 1 for OD file, 0 for TAPE6
- instrument: int, 1=CrIS, 2=IASI
- tape5f: output filenanme (path included)

Output:
- TAPE5 file written to disk as 'tape5f'
"""


import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def lbl_write_tape5(prof, comment, VBOUND, TBOUND, HBOUND, aflag, CO2amt, gcflag, selwn, outfile_type, instrument, tape5f):

    with open(tape5f, "w") as fid:
        fid.write(f"$ {comment} \n")

        if gcflag == 1:
            flagstring = (' HI=1 F4=1 CN=1 AE=0 EM=1 SC=3 FI=0 PL=0 TS=0 AM=1 MG=' +
                          ('1' if outfile_type == 1 else '0') + ' LA=0 OD=0 XS=1   00   00')
        else:
            flagstring = (' HI=1 F4=1 CN=6 AE=0 EM=1 SC=3 FI=0 PL=0 TS=0 AM=1 MG=' +
                          ('1' if outfile_type == 1 else '0') + ' LA=0 OD=0 XS=1   00   00')
        fid.write(flagstring + '\n')

        if gcflag in [2, 3, 4, 5, 6]:
            multipliers = {
                2: [1, 1, 0, 0, 0, 0, 0],
                3: [0, 0, 0, 0, 0, 0, 0],
                4: [1, 1, 0, 1, 0, 0, 0],
                5: [0, 0, 0, 1, 0, 0, 0],
                6: [0, 0, 1, 0, 1, 1, 1]
            }
            fid.write("".join(f"{v:10.3f}" for v in multipliers[gcflag]) + '\n')

        fid.write(f"{VBOUND[0]:10.3f}{VBOUND[1]:10.3f}\n")
        fid.write(f"{TBOUND[0]:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}    l\n")

        N = len(prof['pressure'])
        boundaries = prof['pressure']

        #MODEL, ITYPE, IBMAX = 0, 2, -N
        MODEL, ITYPE, IBMAX = 0, 2, -N
        NOZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS = 1, 1, 7, 1, 0, 0
        RE, HSPACE = 0.0, 120.0
        VBAR = (VBOUND[0] + VBOUND[1]) / 2.0
        CO2MX = CO2amt if isinstance(CO2amt, float) else CO2amt[0]

        fid.write(f"{MODEL:5d}{ITYPE:5d}{IBMAX:5d}{NOZERO:5d}{NOPRNT:5d}{NMOL:5d}{IPUNCH:5d}{IFXTYP:2d} {MUNITS:2d}{RE:10.3f}{HSPACE:10.3f}{VBAR:10.3f}{0:10.3f}\n")
        fid.write(f"{HBOUND[0]:10.3f}{HBOUND[1]:10.3f}{HBOUND[2]:10.3f}\n")

        for i, val in enumerate(boundaries):
            fid.write(f"{val:10.4f}")
            if (i+1) % 8 == 0 and (i+1) != N:
                fid.write('\n')
        fid.write('\n')

        fid.write(f"{IBMAX:5d} levels in the user defined profile\n")

        aflags = ['1111', '2222', '3333', '4444', '5555', '6666']
        aflagstring = aflags[aflag - 1] if 1 <= aflag <= 6 else 'AAAA'

        for i in range(N):
            if gcflag == 1:
                gcvar = [prof['w'][i], CO2amt[i] if hasattr(CO2amt, '__getitem__') else CO2amt, prof['oz'][i]]
            elif gcflag == 2 or gcflag == 3:
                gcvar = [prof['w'][i], 0.0, 0.0]; aflagstring = 'AAAA'
            elif gcflag == 4:
                gcvar = [prof['w'][i], 0.0, prof['oz'][i]]; aflagstring = 'AAAA'
            elif gcflag == 5:
                gcvar = [0.0, 0.0, prof['oz'][i]]; aflagstring = 'AAAA'
            elif gcflag == 6:
                gcvar = [0.0, CO2amt[i] if hasattr(CO2amt, '__getitem__') else CO2amt, 0.0]; aflagstring = 'AAAA'

            #fid.write(f"{0.0:10.3f}{prof['pressure'][i]:10.5f}{prof['tdry'][i]:10.3f}     AA   CAA{aflagstring}\n")
            fid.write(f"{prof['alt'][i]/1000.0:10.3f}{prof['pressure'][i]:10.5f}{prof['tdry'][i]:10.3f}     AA   CAA{aflagstring}\n")
            fid.write("".join(f"{v:10.5f}" for v in gcvar) + '\n')

        fid.write("    3    0    0 selected x-sections are :\n")
        fid.write("CCL4      F11       F12 \n")
        fid.write("    2    1   \n")
        fid.write(f"{boundaries[0]:10.3f}     AAA\n")
        fid.write("1.105e-04 2.783e-04 5.027e-04\n")
        fid.write(f"{boundaries[-1]:10.3f}     AAA\n")
        fid.write("1.105e-04 2.783e-04 5.027e-04\n")

        # FFT parameters and wavenumber grid for spectral output.
        fftmargin = 10.0

        v1 = VBOUND[0] + fftmargin
        v2 = VBOUND[1] - fftmargin

        # --- Begin branch for FFTSCAN parameters ---
        if instrument == 1:  # S-HIS
            string1 = '1.03702766'
            Xmax = float(string1)
            dv = 1.0 / (2 * Xmax)
            j1 = int(v1 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            j2 = int(v2 / dv) + 1
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1   -4    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        elif instrument == 2:  # IASI
            string1 = '2.00000000'
            Xmax = float(string1)
            dv = 1.0 / (2 * Xmax)
            j1 = int(v1 / dv) + 1
            j2 = int(v2 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1   -2    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        elif instrument == 3:  # CrIS: use max OPD = 0.8 cm
            string1 = '0.80000000'  # CrIS maximum OPD in cm
            Xmax = float(string1)
            dv = 1.0 / (2 * Xmax)  # dv in cm^-1; here dv = 0.625 cm^-1
            j1 = int(v1 / dv) + 1
            j2 = int(v2 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            # Set an assumed flag string; adjust as needed for CrIS.
            string4 = '    1   -3    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        else:
            raise ValueError("ERROR: instrument not properly defined")

        out_string = string1 + string2 + string3 + string4 + string5 + string6
        fid.write(out_string)
        fid.write('-1.\n')

       # FFTSCAN block for upwelling radiance and brightness temperature at TOA.
        v1 = v1 + fftmargin
        v2 = v2 - fftmargin

        if instrument == 1:  # S-HIS
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '1.03702766'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        elif instrument == 2:  # IASI
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '2.00000000'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        elif instrument == 3:  # CrIS
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '0.80000000'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        else:
            raise ValueError("ERROR: instrument not properly defined")

        out_string = string1 + string2 + string3 + string4 + string5
        fid.write(out_string)
        fid.write('-1.\n')


        fid.write('$ Transfer to ASCII plotting data\n')
        fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=0 FI=0 PL=1 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
        fid.write('# Plot title not used\n')
        string7 = '   10.2000  100.0000    5    0   11    0    1.0000 0  0    0\n'
        out_string = string2 + string3 + string7
        fid.write(out_string)
        fid.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 27\n')
        fid.write(out_string)
        fid.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 28\n')
        fid.write('-1.\n')
        fid.write('%\n')

    return

def lbl_write_tape5_od(prof, comment, VBOUND, TBOUND, HBOUND, aflag, CO2amt, gcflag, selwn, outfile_type, instrument, tape5f):

    with open(tape5f, "w") as fid:
        fid.write(f"$ {comment} \n")

        flagstring = (' HI=1 F4=1 CN=1 AE=0 EM=1 SC=3 FI=0 PL=0 TS=0 AM=1 MG=0 LA=0 OD=0 XS=1   00   00')
        fid.write(flagstring + '\n')

        if gcflag in [2, 3, 4, 5, 6]:
            multipliers = {
                2: [1, 1, 0, 0, 0, 0, 0],
                3: [0, 0, 0, 0, 0, 0, 0],
                4: [1, 1, 0, 1, 0, 0, 0],
                5: [0, 0, 0, 1, 0, 0, 0],
                6: [0, 0, 1, 0, 1, 1, 1]
            }
            fid.write("".join(f"{v:10.3f}" for v in multipliers[gcflag]) + '\n')

        fid.write(f"{VBOUND[0]:10.3f}{VBOUND[1]:10.3f}\n")
        #fid.write(f"{TBOUND[0]:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}    l\n")
        fid.write(f"{TBOUND[0]:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}{-1:10.3f}{0:10.3f}{0:10.3f}\n")

        N = len(prof['pressure'])
        boundaries = prof['pressure']

        #MODEL, ITYPE, IBMAX = 0, 2, -N
        MODEL, ITYPE, IBMAX = 0, 2, -N
        NOZERO, NOPRNT, NMOL, IPUNCH, IFXTYP, MUNITS = 1, 1, 7, 1, 0, 0
        RE, HSPACE = 0.0, 120.0
        VBAR = (VBOUND[0] + VBOUND[1]) / 2.0
        CO2MX = CO2amt if isinstance(CO2amt, float) else CO2amt[0]

        fid.write(f"{MODEL:5d}{ITYPE:5d}{IBMAX:5d}{NOZERO:5d}{NOPRNT:5d}{NMOL:5d}{IPUNCH:5d}{IFXTYP:2d} {MUNITS:2d}{RE:10.3f}{HSPACE:10.3f}{VBAR:10.3f}{0:10.3f}\n")
        fid.write(f"{HBOUND[0]:10.3f}{HBOUND[1]:10.3f}{HBOUND[2]:10.3f}\n")

        for i, val in enumerate(boundaries):
            fid.write(f"{val:10.4f}")
            if (i+1) % 8 == 0 and (i+1) != N:
                fid.write('\n')
        fid.write('\n')

        fid.write(f"{IBMAX:5d} levels in the user defined profile\n")

        aflags = ['1111', '2222', '3333', '4444', '5555', '6666']
        aflagstring = aflags[aflag - 1] if 1 <= aflag <= 6 else 'AAAA'

        for i in range(N):
            if gcflag == 1:
                gcvar = [prof['w'][i], CO2amt[i] if hasattr(CO2amt, '__getitem__') else CO2amt, prof['oz'][i]]
            elif gcflag == 2 or gcflag == 3:
                gcvar = [prof['w'][i], 0.0, 0.0]; aflagstring = 'AAAA'
            elif gcflag == 4:
                gcvar = [prof['w'][i], 0.0, prof['oz'][i]]; aflagstring = 'AAAA'
            elif gcflag == 5:
                gcvar = [0.0, 0.0, prof['oz'][i]]; aflagstring = 'AAAA'
            elif gcflag == 6:
                gcvar = [0.0, CO2amt[i] if hasattr(CO2amt, '__getitem__') else CO2amt, 0.0]; aflagstring = 'AAAA'

            #fid.write(f"{0.0:10.3f}{prof['pressure'][i]:10.5f}{prof['tdry'][i]:10.3f}     AA   CAA{aflagstring}\n")
            fid.write(f"{prof['alt'][i]/1000.0:10.3f}{prof['pressure'][i]:10.5f}{prof['tdry'][i]:10.3f}     AA   CAA{aflagstring}\n")
            fid.write("".join(f"{v:10.5f}" for v in gcvar) + '\n')
            print(f"w: {gcvar[0]:.6e}, CO2: {gcvar[1]:.6e}, oz: {gcvar[2]:.6e}")

        fid.write("    3    0    0 selected x-sections are :\n")
        fid.write("CCL4      F11       F12 \n")
        fid.write("    2    1   \n")
        fid.write(f"{boundaries[0]:10.3f}     AAA\n")
        fid.write("1.105e-04 2.783e-04 5.027e-04\n")
        fid.write(f"{boundaries[-1]:10.3f}     AAA\n")
        fid.write("1.105e-04 2.783e-04 5.027e-04\n")

        # FFT parameters and wavenumber grid for spectral output.
        fftmargin = 10.0

        v1 = VBOUND[0] + fftmargin
        v2 = VBOUND[1] - fftmargin

        # --- Begin branch for FFTSCAN parameters ---
        if instrument == 1:  # S-HIS
            string1 = '1.03702766'
            Xmax = float(string1)
            dv = 1.0 / (2 * Xmax)
            j1 = int(v1 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            j2 = int(v2 / dv) + 1
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1   -4    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        elif instrument == 2:  # IASI
            string1 = '2.00000000'
            Xmax = float(string1)
            dv = 1.0 / (2.0 * Xmax)
            j1 = int(v1 / dv) + 1
            j2 = int(v2 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1   -2    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        elif instrument == 3:  # CrIS: use max OPD = 0.8 cm
            string1 = '0.80000000'  # CrIS maximum OPD in cm
            Xmax = float(string1)
            dv = 1.0 / (2. * Xmax)  # dv in cm^-1; here dv = 0.625 cm^-1
            j1 = int(v1 / dv) + 1
            j2 = int(v2 / dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            # Set an assumed flag string; adjust as needed for CrIS.
            string4 = '    1   -3    0       0.0   12    1    1   13  0 0  '
            string5 = f'{dv:10.8f}'
            string6 = '\n'
        else:
            raise ValueError("ERROR: instrument not properly defined")

        out_string = string1 + string2 + string3 + string4 + string5 + string6 
        fid.write(out_string)
        fid.write('-1.\n')

       # FFTSCAN block for upwelling radiance and brightness temperature at TOA.
        v1 = v1 + fftmargin
        v2 = v2 - fftmargin

        if instrument == 1:  # S-HIS
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '1.03702766'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        elif instrument == 2:  # IASI
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '2.00000000'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        elif instrument == 3:  # CrIS
            fid.write('$ *** FFTSCAN for the Upwelling Radiance and Brightness Temperature at TOA ***\n')
            fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=2 FI=0 PL=0 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
            string0 = '0.80000000'
            Xmax = float(string0)
            dv = 1.0/(2*Xmax)
            j1  = int(v1/dv) + 1
            j2  = int(v2/dv) + 1
            string2 = f'{(j1-1)*dv:10.5f}'
            string3 = f'{(j2-1)*dv:10.5f}'
            string4 = '    1    1                  13    1    1   11    0'
            string1 = f'{dv:10.8f}'
            string5 = '\n'
        else:
            raise ValueError("ERROR: instrument not properly defined")

        out_string = string1 + string2 + string3 + string4 + string5
        fid.write(out_string)
        fid.write('-1.\n')


        fid.write('$ Transfer to ASCII plotting data\n')
        fid.write(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=0 FI=0 PL=1 TS=0 AM=0 MG=0 LA=0 OD=0 XS=0    0    0\n')
        fid.write('# Plot title not used\n')
        string7 = '   10.2000  100.0000    5    0   11    0    1.0000 0  0    0\n'
        out_string = string2 + string3 + string7
        fid.write(out_string)
        fid.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 27\n')
        fid.write(out_string)
        fid.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 28\n')
        fid.write('-1.\n')
        fid.write('%\n')

    return

"""
def read_tape27(filename, scale=1e7):
    rad = []
    wnum = []
    with open(filename, 'r') as fn:
        for i, line in enumerate(fn):
            if i > 26:
                parsed = line.split()
                if len(parsed) >= 2:
                    wnum.append(parsed[0])
                    rad.append(parsed[1])
    return np.asarray(wnum, dtype=float), np.asarray(rad, dtype=float) * scale
"""


def read_tape27(filename, scale=1e7):
    rad = []
    wnum = []
    spectral_started = False

    with open(filename, 'r') as fn:
        for line in fn:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Look for spectral data start (after header block)
            if not spectral_started:
                if stripped.startswith('0') and 'WAVENUMBER' in stripped and 'RADIANCE' in stripped:
                    spectral_started = True
                continue

            # From here we expect spectral data
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    wnum.append(float(parts[0]))
                    rad.append(float(parts[1]) * scale)
                except ValueError:
                    print(f"[WARN] Could not parse numbers: {stripped}")
            else:
                # Could print for debug, but optional
                # print(f"[WARN] Ignored non-data line: {stripped}")
                continue

    wnum_arr = np.asarray(wnum)
    rad_arr = np.asarray(rad)

    print(f"[INFO] Parsed {len(wnum_arr)} spectral points from {filename}")
    return wnum_arr, rad_arr


def read_tape7_profile(filepath):
    """
    Extracts pressure, temperature and height from the first 101 levels of a TAPE7 file.

    Parameters:
        filepath (str): Path to the TAPE7 file

    Returns:
        z (list): Geopotential height [km]
        p (list): Pressure [hPa]
        T (list): Temperature [K]
    """
    z, p, T = [], [], []

    with open(filepath, 'r') as f:
        f.readline()  # skip header
        header = f.readline()
        n_layers = int(header[2:5])

        for i in range(n_layers):
            line1 = f.readline()
            f.readline()  # skip second line (molecular concentrations)

            floats = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", line1)

            if i == 0:
                if len(floats) >= 3:
                    try:
                        z.append(float(floats[2]))
                        p.append(float(floats[3]))
                        T.append(float(floats[4]))
                    except Exception:
                        print(f"[WARN] Failed to parse first layer: {line1.strip()}")
                else:
                    print(f"[WARN] First layer: not enough floats found: {line1.strip()}")
            else:
                if len(floats) >= 3:
                    try:
                        z.append(float(floats[-3]))
                        p.append(float(floats[-2]))
                        T.append(float(floats[-1]))
                    except Exception:
                        print(f"[WARN] Failed to parse layer {i}: {line1.strip()}")
                else:
                    print(f"[WARN] Layer {i}: not enough floats found: {line1.strip()}")
    for i in range(len(z)):
        print(f"{i:3d}  z = {z[i]:8.3f} km   p = {p[i]:9.4f} hPa   T = {T[i]:7.2f} K")
    return z, p, T

def read_lbl_outputs(instrument, profile_dir):
    print("[INFO] Reading outputs...")
    
    print(f"Type of profile_dir: {type(profile_dir)} value: {profile_dir}")

    # If profile_dir comes in as string, convert to Path for convenience
    profile_dir = Path(profile_dir)

    # DEBUG: List all files in the directory
    print(f"DEBUG: Listing contents of {profile_dir}")
    for f in profile_dir.iterdir():
        print(f" - {f.name}")

    # Check for TAPE27 specifically
    tape27 = profile_dir / 'TAPE27'
    if not tape27.exists():
        raise FileNotFoundError(f"TAPE27 not found at {tape27}")

    print(f"DEBUG: About to read {tape27}")
    print(f"DEBUG: Exists? {tape27.exists()}")

    w27, r27 = read_tape27(tape27)

    if instrument==1:
        instrument_str='S-HIS'
    elif instrument==2:
        instrument_str='IASI'
    elif instrument==3:
        instrument_str='CrIS'
    else:
        print(f'Insturument {instrument} Unknown Istrument, use 1 for S-HIS, 2 for IASI, 3 for CrIS')
        return -1

    return w27, r27, instrument_str

def read_outputs_and_plot(instrument, tape7_path, tape27_path):
    print("[INFO] Reading and plotting outputs...")

    z, p, T  = read_tape7_profile(tape7_path)
    w27, r27 = read_tape27(tape27_path)

    if instrument==1:
        instrument_str='S-HIS'
    elif instrument==2:
        instrument_str='IASI'
    elif instrument==3:
        instrument_str='CrIS'
    else:
        print(f'Insturument {instrument} Unknown Istrument, use 1 for S-HIS, 2 for IASI, 3 for CrIS')
        return -1


    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    ax1.plot(w27, r27, label=f'{instrument_str} Radiance', linewidth=0.75, color='blue')
    ax1.set_xlabel('Wavenumber [cm⁻¹]')
    ax1.set_ylabel('Radiance [mW/(m²·sr·cm⁻¹)]')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    for i in range(len(z)):
        print(f"{i:3d}  z = {z[i]:8.3f} km   p = {p[i]:9.4f} hPa   T = {T[i]:7.2f} K")

    ax2 = ax1.inset_axes([0.65, 0.55, 0.3, 0.3])
    ax2.plot(T, z, linewidth=1.2, color='black')
    ax2.set_title('Temperature profile')
    ax2.set_xlabel('T [K]')
    ax2.set_ylabel('z [km]')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'spectrum_with_profile_{instrument_str}.png')
    plt.close()

    return
