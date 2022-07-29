#!/usr/bin/env python3
import os
import process_attr as pa

#----------------------------------------------------------------------------

def main():
    attr_list = pa.load_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attr_data/attr_list.pkl'))
    for i, item in enumerate(attr_list):
        print(f"{i}: {item}")

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
