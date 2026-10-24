# Portraits kept out of the sealed draw pool

`targets_img/photos/` is the gallery the sealed photo run draws from. Portraits
here are candidates with their credit; move one into `photos/` to add it.

## bose.png

Jagadish Chandra Bose, 1926, Agence de presse Meurisse, public domain
(Wikimedia Commons, `File:Jagadish Chandra Bose 1926.jpg`). Crop in
`make_photo_gallery.CROPS["bose"]`. At two grey levels the face and the suit
fall on the same side of the threshold and the 12 x 12 target is a dark
silhouette; at four levels on 16 x 16 the hair, face and suit separate. Use
him only with the four-level gallery, which the wedge run has to justify.

## Colourised portraits, for the colour runs

The colours are a colourist's, not the sitter's; the paper must say so.
Brightness and colour targets come from `make_colour_pair.py`, crops are
`CROPS["<name>_c"]`.

- `lincoln_colourised.jpg`: Alexander Gardner's November 1863 portrait,
  colourised; public domain per Wikimedia Commons
  (`File:Abraham Lincoln (1809-1865) - Daguerréotype de Lincoln par Alexander Gardner en 1863 (colorisé).jpg`).
- `einstein_colourised.jpg`: colourised by DonkeyHotey, CC BY 2.0
  (`File:Albert Einstein - Colorized.jpg`, from flickr.com/photos/donkeyhotey/12637209434).
- `tesla_colourised.png`: Tesla at 40, coloured by Ivar van Wooning, CC BY-SA 3.0
  (`File:Nikola Tesla Colored.png`).
- Mann: the colour studio headshot stays outside the repo
  (`C:\Users\alexa\Downloads\SteveMannheadshot.jpg`); only its targets are committed.
