
//#region Pacchetti
    // https://typst.app/universe/package/polylux
    #import "@preview/polylux:0.4.0": *
    // #import "@preview/touying:0.6.1": *

    #import "@preview/lovelace:0.3.0": *
//#endregion

//#region Tipografiche
    #let title-size = 32pt
    #let sub-title-size = 30pt
    #let section-size = 30pt
    #let text-size = 24pt

    // https://typst.app/docs/tutorial/making-a-template/
    #let mainTemplate(body) = {
        set page(paper:"presentation-4-3")
        set par(justify: true)

        // Riproduzione dei fonti di ArsClassica
        set text(font:"TeX Gyre Pagella", size:text-size) // Con grazie
        show heading: set text(font:"Iwona", weight:"light") // Senza grazie
        show math.equation: set text(font:"Euler Math") // Matematico

        body
    }
//#endregion

//#region Matematica
    #let mathTemplate(body) = {
        // https://typst.app/docs/reference/layout/h/#math-spacing
        show math.plus: it => $#h(.1em) it #h(.1em)$
        show math.minus: it => $#h(.1em) it #h(.1em)$
        // show math.times: it => $#h(-.1em) it #h(-.1em)$

        show math.eq: it => $#h(.2em) it #h(.2em)$
        // show math.lt: it => $#h(-.1em) it #h(-.1em)$
        // show math.gt: it => $#h(-.1em) it #h(-.1em)$

        body
    }

    #let de = $dif$
    #let RPlus = $RR_+$
    #let pderv(var) = $frac(de, de var)$

    #let underhat(body) = math.attach(
        math.limits(body),
        b:move(dy:-0.1em, math.hat),
    )
//#endregion

//#region Colori
    #let CTred = rgb("#800000") // Colore rosso di ClassicThesis
//#endregion

//#region Funzioni
    // https://typst.app/docs/reference/introspection/state/
    #let section-title = state("section-title","")
    #let section-subtitle = state("section-subtitle","")

    // https://typst.app/docs/reference/foundations/function/
    // https://typst.app/docs/reference/scripting/#bindings
    #let section(body,title:"") = [
        #section-title.update(title)
        #section-subtitle.update("")
        // Si resetta il sottotitolo quando si cambia sezione

        #set page(
            // paper: "presentation-16-9",
            margin: (x:2em,top:3em,bottom:1em),
            header: context{
                let t = section-title.get()
                let s = section-subtitle.get()
                let hsep = .75em
                let sep = [#h(hsep)\-#h(hsep)]

                align(
                    top,
                    toolbox.full-width-block(
                        fill:white,
                        inset: (x:1em,y:.75em)
                    )[
                        #set text(
                            size: section-size,
                            fill: CTred
                        )
                        #t#if s != "" [#sep#s]
                    ]
                )
            }
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

    #let subsection(body,subtitle:"") = [
        #section-subtitle.update(subtitle)
        #body
    ]
//#endregion
