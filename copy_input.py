import filecmp, os

for line in open("source.list"):
    if len(line)==0:
        continue
    else:
        value = line.split('#')[0]
        if len(value) == 0:
            continue
        else:
            parameters = value.split()

            if len(parameters) in [2, 3]:
                infile  = os.path.expanduser(parameters[0])

                if len (parameters) == 2:
                    outfile = parameters[1].rstrip('/')\
                        +'/'+infile.split('/')[-1]
                else:
                    outfile = parameters[1].rstrip('/')+'/'+parameters[2]
                
                if not os.path.exists(outfile):
                    os.system('cp {0} {1}'.format(infile, outfile))
                    print "Added file:   {0}".format(outfile)
                else:
                    if not filecmp.cmp(infile, outfile):
                        os.system('cp {0} {1}'.format(infile, outfile))
                        print "Updated file: {0}".format(outfile)

                
            
            
