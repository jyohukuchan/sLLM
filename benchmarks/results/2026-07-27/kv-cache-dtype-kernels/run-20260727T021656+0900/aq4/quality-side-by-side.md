# Long-context KV-cache dtype generation

The same 3,968-token input was used for all rows. FP8 is OCP E4M3FN/S1E4M3.

## F32

Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing multiple "Notebook block" entries (000 to 043, then 254, 255).
    *   Content of each block: Describes archive cards

Token IDs: `90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262,348,256,5396,25,357,1248,1414,8222,5081,328,8839,2121,2424,1,10379,318,15,15,15,310,220,15,19,18,11,1179,220,17,20,19,11,220,17,20,20,553,198,262,348,256,8615,314,1754,2424,25,3744,53283,17593,7176`

## F16

Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing multiple "Notebook block" entries (000 to 043, then 254, 255).
    *   Content of each block: Describes archive cards

Token IDs: `90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262,348,256,5396,25,357,1248,1414,8222,5081,328,8839,2121,2424,1,10379,318,15,15,15,310,220,15,19,18,11,1179,220,17,20,19,11,220,17,20,20,553,198,262,348,256,8615,314,1754,2424,25,3744,53283,17593,7176`

## FP8_E4M3FN

Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing multiple "Notebook block" entries (000 to 043, then 254, 255).
    *   Content of each block: Describes archive cards

Token IDs: `90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262,348,256,5396,25,357,1248,1414,8222,5081,328,8839,2121,2424,1,10379,318,15,15,15,310,220,15,19,18,11,1179,220,17,20,19,11,220,17,20,20,553,198,262,348,256,8615,314,1754,2424,25,3744,53283,17593,7176`
