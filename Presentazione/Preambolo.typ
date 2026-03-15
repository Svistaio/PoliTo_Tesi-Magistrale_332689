
// https://typst.app/docs/guides/for-latex-users/

//#region Pacchetti
    #import "@preview/lovelace:0.3.0":*
    #import "@preview/cetz:0.4.2":*
    // #import "@preview/fletcher:0.5.8":* // https://typst.app/universe/package/fletcher/
    // Il pacchetto «fletcher» non ha ancora la semplice chiave dell'ancora per i suoi nodi, cosa che rende il posizionamento relativo alquanto noioso da gestire

    // Mi sono reso conto che Typst è sufficientemente flessibile da non necessitare di particolari pacchetti per creare una presentazione
    // #import "@preview/polylux:0.4.0":*// https://typst.app/universe/package/polylux
    // #import "@preview/touying:0.6.1":*
//#endregion

//#region Impostazioni
  //#region Bozza
    #let draft = true
    #let draft = false

    #let draftColour(clr) = if draft {clr} else {white}
    #let draftStroke(
      thickness:1pt,colour:black
    ) = if draft {thickness+colour} else {none}
    #let draftFill(colour:black) = if draft {colour} else {none}
  //#endregion

  //#region Tipografia
    // https://typst.app/docs/tutorial/making-a-template/
    //#region Modello diapo
      #let slide-sizes = (
        title:32pt,
        sub-title:30pt,
        section:30pt,
        text:24pt,
      )
  
      #let slideTemplate(body) = {
        set page(
          paper:"presentation-4-3"
          // paper: "presentation-16-9",
        )
        set par(justify:true)
  
        // Riproduzione dei fonti di ArsClassica
        set text(
          size:slide-sizes.text,
          font:"TeX Gyre Pagella",
          lang:"it",
        ) // Con grazie
  
        show heading: set text(
          font:"Iwona",
          weight:"light"
        ) // Senza grazie
  
        show math.equation: set text(
          font:"Euler Math"
        ) // Matematico
  
        let list-indent = 1em
        // https://typst.app/docs/reference/model/list/
        set list(
          marker:[
            #box(height:.8em)[
              #align(horizon)[
                #rotate(45deg)[
                  #square(
                    width:7pt,
                    fill:white,
                    stroke:1pt
                  )
                ]
              ]
            ]
          ],
          indent:list-indent,
        )
  
        // https://typst.app/docs/reference/model/enum/
        set enum(indent:list-indent)
  
        body
      }
    //#endregion

    //#region Modello articolo
      #let article-sizes = (
        title:18pt,
        sub-title:15pt,
        section:13pt,
        text:11pt,
      )
  
      #let articleTemplate(body) = {
        set page(
          paper:"a4",
          margin:(x:4cm,y:3cm),
          background:rect(
            width:100%,
            height:100%,
            fill:draftColour(red.lighten(70%)),
          ),
          numbering:"1"
        )// https://typst.app/docs/guides/page-setup/
        set par(
          justify:true,
          // first-line-indent:1.5em,
          // spacing:.65em,
        )
  
        // Riproduzione dei fonti di ArsClassica
        set text(
          size:article-sizes.text,
          font:"TeX Gyre Pagella",
          lang:"it",
        ) // Con grazie
  
        show heading: set text(
          font:"Iwona",
          weight:"light"
        ) // Senza grazie
  
        show math.equation: set text(
          font:"Euler Math"
        ) // Matematico
  
        let list-indent = 1em
        // https://typst.app/docs/reference/model/list/
        set list(
          marker:[
            #box(height:.8em)[
              #align(horizon)[
                #rotate(45deg)[
                  #square(
                    width:7pt,
                    fill:white,
                    stroke:1pt
                  )
                ]
              ]
            ]
          ],
          indent:list-indent,
        )
  
        // https://typst.app/docs/reference/model/enum/
        set enum(indent:list-indent)
  
        block(
          width:100%,
          // height:100%, // Se si vuole che il testo sia separato in piú pagine bisogna commentare l'altezza essendo un vincolo eccessivamente rigido
          fill:draftColour(blue.lighten(70%)),
          inset:0pt,
          breakable:true,
        )[
          #set align(left+top)
          #body
        ]
      }
    //#endregion
  //#endregion

  //#region Immagini
    #let border = 0.5pt+black
    #let rowHeight = 6.5cm

    #let factor(number,exponent,base:1) = {base+number*calc.pow(10,exponent)}
    #let img-height-14 = rowHeight*factor(1.75,-1)
    #let img-height-23 = rowHeight*factor(1.75,-1)
    #let img-dy(shift:0cm) = -19.1cm+shift
  //#endregion

  //#region Colori
    #let CTred = rgb("#800000") // Colore rosso di ClassicThesis
    // https://typst.app/docs/reference/visualize/color/
  //#endregion
//#endregion

//#region Suddivisione
  // https://typst.app/docs/reference/introspection/state/
  #let section-title = state("section-title","")
  #let section-subtitle = state("section-subtitle","")
  
  // https://polylux.dev/book/toolbox/progress.html
  #let progress = context {
    let current = counter(page).at(here()).first()
    let final = counter(page).final().first()
    [#current/#final]
  }
  
  #let section(body,title:"") = [
    // Si resetta il sottotitolo quando si cambia sezione
    #section-title.update(title)
    #section-subtitle.update("")
    #body
  ]
  
  #let subsection(body,subtitle:"") = [
    #section-subtitle.update(subtitle)
    #body
  ]
  
  #let default-x-margin = 2em
  #let default-top-margin = 1em
  #let default-bottom-margin = 1em

  #let slide(
    body,
    width:28cm,
    height:none,
    margin:(
      x:default-x-margin,
      top:default-top-margin,
      // bottom:default-bottom-margin
    ),
    header-alignment:left,
    footer-alignment:right,
    // alignment:top,
    alignment:horizon,
    header:true,
    footer:true,
    extremes:false,
    centred-content:true,
  ) = {
    context {
      let height = if height == none {width*3/4} else {height}

      // Inizializzazione
      let header-height = 0pt
      let footer-height = 0pt
      let header-inner = none
      let footer-inner = none
      let default-bottom-margin = default-bottom-margin
      
      if extremes {
        if header {header-height = 0pt}
        if footer {footer-height = 0pt}
      } else {
        if header {
          header-inner = {
            let t = section-title.get()
            let s = section-subtitle.get()
            let hsep = .5em
            let sep = [#h(hsep)\-#h(hsep)]
            
            block(
              width:width,
              fill:draftColour(black.lighten(80%)),
              inset:15pt,
            )[
              #set text(size:slide-sizes.section,fill:CTred)
              #align(header-alignment)[#t#if s != "" [#sep#s]]
            ]
          }
        }

        if footer {
          footer-inner = {
            block(
              width:width,
              fill:draftColour(black.lighten(80%)),
              inset:8pt,
            )[
              #set text(size:slide-sizes.text,fill:CTred)
              #align(footer-alignment)[#progress]
            ]
          }
        }

        // Misurazione dinamica dell'altezza
        header-height = measure(header-inner).height
        footer-height = measure(footer-inner).height

        if centred-content {
          default-bottom-margin = header-height + default-top-margin - footer-height
        } // Se si vuole il contenuto centrato si calcola il margine inferiore per tale scopo
      }

      let side-margin = margin.at("x",default:default-x-margin)
      let top-page = if header {
        header-height+margin.at("top",default:default-top-margin)
      } else {margin.at("top",default:0em)}
      let bottom-page = if footer {
        footer-height+margin.at("bottom",default:default-bottom-margin)
      } else {margin.at("bottom",default:0em)}
      // let side-margin = margin.x
      // let top-page = header-height+margin.top
      // let bottom-page = footer-height+margin.bottom

      // https://typst.app/docs/reference/layout/page/
      page(
        width:width,
        height:height,
        margin:(
          x:side-margin,
          top:top-page,
          bottom:bottom-page
        ),
        header-ascent:0pt,
        footer-descent:0pt,
        header: if not extremes and header{
          align(top)[
            #move(dx:-side-margin)[
              #block(
                width:width,
                height:header-height,
                stroke: draftStroke(),
                inset:0pt,
              )[#header-inner]
            ]
          ]
        },
        footer: if not extremes and footer{
          align(bottom)[
            #move(dx:-side-margin)[
              #block(
                width:width,
                height:footer-height,
                stroke: draftStroke(),
                inset:0pt,
              )[#footer-inner]
            ]
          ]
        },
        background:rect(
          width:100%,
          height:100%,
          fill:draftColour(red.lighten(70%)),
        ),
      )[
        #block(
          width:100%,
          height:100%,
          fill:draftColour(blue.lighten(70%)),
          inset:0pt,
        )[
          #set align(alignment)
          #body
        ]
      ]
    }
  }
//#endregion

//#region Matematica
  #let mathEnvTemplate(body) = {
    // https://typst.app/docs/reference/layout/h/#math-spacing
    show math.plus: it => $#h(.1em) it #h(.1em)$
    show math.minus: it => $#h(.1em) it #h(.1em)$
    // show math.times: it => $#h(-.1em) it #h(-.1em)$

    show math.eq: it => $#h(.2em) it #h(.2em)$
    // show math.lt: it => $#h(-.1em) it #h(-.1em)$
    // show math.gt: it => $#h(-.1em) it #h(-.1em)$

    set math.cases(gap:.5em)

    body
  }

  #let bis(shift:-.4em) = h(shift) // «Before Integral Shift»

  #let de = $dif$
  #let RPlus = $RR_+$
  #let pderv(var) = $frac(de,de var)$

  #let underhat(body,subscript:none) = math.attach(
    math.limits(body),
    b:move(dy:-0.1em, math.hat),
    br:subscript,
  )// https://typst.app/docs/reference/math/attach/

  #let BracketRef(body,subscript) = [$[#text(font:"TeX Gyre Pagella")[#body]]_#text(font:"TeX Gyre Pagella")[#subscript]$]

  #let num(number,e:6,sep:-1pt) = {$number#h(sep)times#h(sep)10^#e$}

  #let pzclR = text(font:"cmsy10")[R]
  #let pzclS = text(font:"cmsy10")[S]
  // set text(font:"jsMath-cmsy10")

  #let RPlus = math.attach(math.bb("R"),b:math.plus)
  #let NPlus = math.attach(math.bb("N"),b:math.plus)

  #let dstr(name) = text(font:"TeX Gyre Pagella")[#name]
  #let DstrBernoulli = dstr("Bernoulli")
  #let DstrGamma =dstr("Gamma")

  #let comment(it,fill:true) = [
      #set text(size:14pt,fill:gray)
      #if fill == true [#h(1fr)] \# #it
  ]

  #let pzclSsup(superscript,shift:.1em) = {
    math.attach(
      pzclS,
      t:[#h(shift)#superscript]
    )
  }

  #let boxed(
    body,
    text:"",
    offset:15pt,
    blk-args:(),
    box-args:(),
  ) = {
    block(
      below:1em,
      ..blk-args,
    )[
        #box(
          width:100%,
          inset:(
            x:10pt,
            top:if text == "" {10pt} else {offset},
            bottom:10pt
          ),
          radius:5pt,
          stroke:1pt+black,
          ..box-args,
        )[
          #if text != "" {
            place(
              top+left,
              dx:1.5em,
              dy:-11pt-offset,
              block(
                fill:white,
                inset:(x:2pt)
              )[#text]
            )
          }#body
        ]
    ]
  }
//#endregion

//#region Immagini
  // https://typst.app/docs/reference/scripting/#loops
  
  #let placeimage(
    path,
    width:100%,
    height:1cm,
    dx:0pt,
    dy:0pt,
  ) = { 
    layout(size => context {
      // https://typst.app/docs/reference/visualize/image
      let img = image(path,width:1cm,fit:"contain")
      
      // Misura l'immagine cosí da ricavare dal contesto sia la sua larghezza che la sua altezza cosí da preservarne il rapporto d'aspetto
      let dim = measure(img)
      let aspect-ratio = dim.height/dim.width
      let linewidth = size.width
      
      let img-width = linewidth
      let img-height = img-width*aspect-ratio
      let blk-height = if height == auto {img-height} else {height}
      
      block(
        width:width,
        height:blk-height,
        clip:true,
        stroke:draftStroke(thickness:0.5pt),
      )[
        #place(
          top+center,
          dx:dx,
          dy:dy,
          image(
            path,
            width:img-width,
            height:img-height,
          ),
        )
      ]
    })
  }

  #let includefigure(
    figure-dict-list,
    width:110%,
    draft:draft,
  ) = [

    #if type(figure-dict-list) == dictionary {
      figure-dict-list = (figure-dict-list,)
    } // Nel caso sia un solo dizionario bisogna renderlo una lista prima di processarlo 
    // Probabilmente vi sono modi migliori d'implementarlo ma per il momento non ho né molta voglia né esperienza per farlo

    #figure(
      block(
        width:width,
        height:auto,
        clip:true,
        stroke:if draft {draftStroke(thickness:0.5pt)},
        [
          #stack(
            dir:ttb,
            spacing:0pt,
            // https://typst.app/docs/reference/foundations/arguments/#spreading
            ..for r in figure-dict-list {
              (placeimage(
                r.path,
                width:r.at("width",default:100%),
                height:r.at("height",default:auto),
                dx:r.at("dx",default:0em),
                dy:r.at("dy",default:0em),
              ),) // Le parentesi tonde servono per proteggere la virgola permettendo a Tyspt di dare un numero arbitrario di argomenti alla funzione «#stack()»
            },
          )
        ],
      ),
      // caption: [Didascalia],
    ) //<figEtichetta>
  ]
//#endregion

//#region Tabelle
  #let padded-row(
    content,
    pos:top,
    pad:.5em,
  ) = content.map(
    c => table.cell(inset:((repr(pos)):pad),c)
  )
//#endregion

//#region Giudizio
  // Apparentemente Typst non ha ancora implementato «abovedisplayshortskip» e «belowdisplayshortskip» per regolare lo spazio tra un equazione e del testo che sborda lievemente in una nuova riga
  // https://typst.app/docs/reference/math/

  // Inoltre non esiste nulla di perfettamente equivalente a 
  // \setlength\thinmuskip{1mu}  % Spazitura sottile
  // \setlength\medmuskip{2mu}   % Spazitura media
  // \setlength\thickmuskip{3mu} % Spazitura spessa
  // Il massimo che si può fare è regolare «manualmente» lo spazio intorno «a ogni singolo operatore» come fatto nello stile matematico sopra
  // show math.plus: it => $#h(.1em) it #h(.1em)$
  // show math.minus: it => $#h(.1em) it #h(.1em)$

  // Molti pacchetti o non mi soddisfano (polylux) o non sono mantenuti da diversi mesi (metro), ma questo è piú un problema di tempo nonostante Typst sia in giro già da qualche anno

  // Alla lista aggiungo anche CeTZ che dovrebb'essere la sostituzione di TikZ ma è ancora estremamente immaturo e a tratti controintuitivo; per esempio, per qualche motivazione non hanno ancora riprodotto la macro «\node» che è alla base di TikZ

  // Tutto sommato direi che Typst va molto bene per le presentazioni (di solito brevi e pieno d'immagini) e per i documenti non ufficiali [non matematici], perciò lo continuerò a usare in quei contesti; invece per i documenti piú ufficiali lo posso usare sse sono soddisfatte due condizioni:
  //   - non vi sono formule matematiche complesse (soprattutto a blocchi e non in linea) e
  //   - non si ricerca uno stile eccessivamente raffinato

  // È tuttavia probabile che migliori in futuro e sono molto curioso di sapere se considereranno anche i dettagli tipografici [che la stragrande maggioranza ignora] oppure se si accontenteranno tralasciandoli
//#endregion
