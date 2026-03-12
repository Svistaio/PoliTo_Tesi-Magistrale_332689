
//#region Preambolo
    #import "Preambolo.typ":*

    #show:articleTemplate
    #show:mathEnvTemplate
//#endregion

//

//#region Note
  #block(
    width:100%,
    height:1cm,
    inset:5pt,
    stroke:draftStroke(),
    below:2em,
  )[
    #set text(size:article-sizes.title)
    #align(center+horizon)[
      _Appunti della presentazione_
    ]
  ]

  #boxed(
    blk-args:(
      below:2em,
    ),
    box-args:(
      inset:5pt
    ),
  )[
    // #set text(size:article-sizes.section)
    #set par(leading: 4pt)
    Durante la presentazione molte informazioni non sono dette e vanno complementate o a voce o alla lavagna; per avere un riferimento è utile scrivere una lista che riassume quanto va detto.
  ]

    #set enum(numbering:"[1]")
    + Diapo iniziale: benvenuto e lettura del titolo.
    + 
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    +
    + Diapo finale: ringraziamenti.
//#endregion
