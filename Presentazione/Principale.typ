
// LTeX: language=it

//#region Preambolo
    //#region Pacchetti
        #import "@preview/polylux:0.4.0": *
        // #import "@preview/touying:0.6.1": *
    //#endregion

    //#region Dimensioni tipografiche
        #let titleSize = 32pt
        #let subTitleSize = 30pt
        #let sectionSize = 32pt
        #let textSize = 25pt
    //#endregion

    //#region Impostazioni
        #set page(paper:"presentation-4-3")

        #set par(justify: true)

        #set text(
            font: "TeX Gyre Pagella",
            size: textSize
        ) // Stesso fonte con grazie del testo di ArsClassica
        
        #show heading: set text(
            font: "Iwona",
            weight: "light"
        ) // Stesso fonte senza grazie del testo di ArsClassica
        
        #show math.equation: set text(
            font: "Euler Math"
        ) // Stesso fonte matematico di ArsClassica
    //#endregion
//#endregion

//#region Funzioni e comandi
    #let CTred = rgb("#800000") // Colore rosso di ClassicThesis

    // https://typst.app/docs/reference/foundations/function/
    // https://typst.app/docs/reference/scripting/#bindings
    #let setSection(body,title:"",subtitle:"") = [
        #set page(
            // paper: "presentation-16-9",
            margin: (x:2em,top:3em,bottom:1em),
            header: align(
                top,
                toolbox.full-width-block(
                    fill:white,
                    inset: (x:1em,y:.75em)
                )[
                    #set text(
                        size: sectionSize,
                        fill: CTred
                    )
                    #let hsep = .75em
                    #let sep = [#h(hsep)\-#h(hsep)]
                    #title#if subtitle != "" [#sep#subtitle]
                ]
            ),
            // footer: align(
            //     bottom,
            //     toolbox.full-width-block(
            //         fill:lime,
            //         inset:8pt
            //     )[I'm down low]
            // ),
        )
        #body
    ]

    // #let setSection(body) = [
    //     #set page(paper: "a4")
    //     #body
    // ]
//#endregion

///

//#region Presentazione
    
    #slide[// Frontespizio
        #set align(center+horizon)

        #text(
            fill:CTred,
            weight:"bold",
            size: titleSize
        )[
                Modellizzazione della distribuzione della popolazione tra città su reti spaziali mediante la teoria cinetica dei sistemi multiagente
        ]

        #v(1em)

        #text(
            fill:CTred,
            style:"italic",
            size: subTitleSize
        )[Corso in Ingegneria Matematica]

        #v(1em)

        Valerio Taralli

        #v(.25em)

        13/03/2026
    ]

    #setSection(
        title:"Introduzione",
        subtitle:"Prova"
    )[
        #slide[
            Questa è una diapo introduttiva per vedere se Typst può introdurre correttamente le immagini:
    
            #figure(
                block( // Si possono anche scrivere i centrimetri
                    width:60%,
                    height:50%,
                    clip:false,
                    // clip:true,
                    place(
                        top+left,
                        image(
                            "../Figure/.fnl/TikZpdf/figTransizioneAsimmetria.pdf",
                            width:100%,
                            height: 100%
                        )
                    )
                ),
                caption: [Porzione specifica del grafico],
            ) //<figProva>
        ]

        #slide[
            Questa è una seconda diapo per mostrare che è possibile averne una seconda collo stesso titolo di sezione.
        ]
    ]
//#endregion
