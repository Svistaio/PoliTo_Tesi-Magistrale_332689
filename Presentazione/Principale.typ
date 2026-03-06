
//#region Preambolo
    #import "Preambolo.typ": * // Include i pacchetti
    // #include "Preambolo.typ" // Non include i pacchetti
    #show: mainTemplate
    #show: mathTemplate
//#endregion

///

//#region Presentazione
    //#region Frontespizio
        #slide[
            #set page(margin: (x:4em,top:2em,bottom:2em))
            #set align(center+horizon)
    
            #v(1fr)
    
            #block(
                width:120%,
                // stroke: 0.5pt+CTred
            )[
                #text(
                    fill:CTred,
                    weight:"bold",
                    size: title-size,
                    // stroke: 0.5pt+CTred
                )[
                    #set par(
                        justify: false,
                        leading: 0.4em,
                    )

                    Modellizzazione della distribuzione della popolazione tra città su reti spaziali mediante la teoria cinetica dei sistemi multiagente
                ]
            ]
    
            #v(1fr)
    
            #text(
                fill:CTred,
                style:"italic",
                size: sub-title-size
            )[Corso in Ingegneria Matematica]

            #v(1fr)

            #block(
                width:100%,
                height:35%,
                clip:true,
                // stroke: 0.5pt+black
            )[
                #place(
                    center+horizon,
                    image(
                        "../Figure/.fnl/Frontespizio/LogoPoliTO.jpg",
                        width:7cm,
                        height:auto
                    ),
                )
            ]

            #v(1fr)
    
            #strong[Relatore]#h(1fr)#strong[Candidato]\
            Andrea Tosin#h(1fr)Valerio Taralli
    
            #v(1fr)
    
            // #v(.25em)
            // 18/03/2026
        ]
    //#endregion

    //#region Introduzione
        #section(title:"Introduzione")[
            #subsection(subtitle:"Figure")[
                #slide[
                    Questa è una diapo introduttiva per vedere se Typst può introdurre correttamente le immagini:
            
                    #layout(size => context {
                        // https://typst.app/docs/reference/visualize/image
                        let img-path = "../Figure/.fnl/TikZpdf/figStudioParametricolRETFSVsD.pdf"
                        let img = image(img-path,width:1cm,fit:"contain")

                        let dim = measure(img)
                        let aspect-ratio = dim.height/dim.width
                        let linewidth = size.width

                        let img-width = linewidth*1.05
                        let img-height = img-width*aspect-ratio

                        figure(
                            block( // Si possono anche scrivere i centrimetri
                                width:110%,
                                height:auto,
                                clip:true,
                                // clip:false,
                                stroke: 0.5pt+black,[
                                    #stack(
                                        dir:ttb,
                                        spacing:0pt,
                                        block(
                                            width:100%,
                                            height:6.2cm*2,
                                            clip: true,
                                            stroke: 0.5pt+black
                                        )[
                                            #place(
                                                top + center,
                                                dy: -000%,
                                                image(
                                                    img-path,
                                                    width:img-width,
                                                    height:img-height
                                                ),
                                            )
                                        ],
                                        block(
                                            width:100%,
                                            height:0.4cm,
                                            clip:true,
                                            stroke: 0.5pt+black
                                        )[
                                            #place(
                                                bottom+center,
                                                dy: 2pt,
                                                image(
                                                    img-path,
                                                    width:img-width,
                                                    height:img-height
                                                ),
                                            )
                                        ]
                                    )
                                ],
                            ),
                            // caption: [Didascalia],
                        ) //<figProva>
                    })
                ]
            ]
            #subsection(subtitle:"Equazioni")[
                #slide[
                    $
                    pderv(t)integral_(cal(I)) integral_(RPlus) Phi f de v de i =
                    integral_(cal(I)^2) integral_(RPlus^2) A(i, i_*) 
                    frac(chevron.l Phi' + Phi'_* - Phi - Phi_* chevron.r, 2) 
                    f f_* de s de s_* de i de i_*
                    $

                    $ underhat(a) underhat(b) underhat(c) underhat(d) underhat(e) underhat(f) underhat(g) underhat(h) underhat(i) underhat(j) underhat(k) underhat(l) underhat(m) underhat(n) underhat(o) underhat(p) underhat(q) underhat(r) underhat(s) underhat(t) underhat(y) underhat(x) underhat(z) $
                    // https://typst.app/docs/reference/math/attach/
                ]
            ]
        ]
    //#endregion

    //#region Simulazioni
        #section(title:"Simulazioni")[
            #subsection(subtitle:"Algoritmo")[
                // https://a5s.eu/toot-lovelace/line-numbers/line-numbers.html
                #slide[
                    #set page(
                        // width:28cm,
                        height:22.5cm,
                        margin: (x:4em,top:2em,bottom:0em),
                        // margin: (x:2em,top:3em,bottom:1em),
                    )

                    #set text(size: 20pt)
                    // #set page(margin:(x:2em, top:3em, bottom:1em),)


                    #let dstr(name) = math.upright(name)
                    #let dstr(name) = math.upright(name)
                    #let comment(it,fill:true) = [
                        #set text(size: 14pt,fill: gray)
                        #if fill == true [#h(1fr)] \# #it
                    ]

                    #let scl = 90%
                    #scale(x:scl,y:scl,origin:center+horizon)[
                        #pseudocode-list(
                            title:none,
                            stroke:0.5pt+black,
                            booktabs:true,
                        )[
                            - *Dati:* $N in NN^+$, $Delta t <= 1$, $sigma$, $T > 0$, $P$, $bold(A)$ e $bold(B)$
                            + $cal(S)^0 <- (s_1^0, s_2^0, ..., s_N^0) equiv (P/N) bold(1) in RR_(>=0)^N$ <algMonteCarloDistribuzioneIniziale>
                            + *per* $n = 0, 1, 2, ..., floor(T / Delta t) - 1$ *fai*
                                + $P <-$ permutazione indipendente di $\{1, 2, ..., N\}$
                                + *per* $i = 1, 2, ..., floor(N / 2)$ *fai*
                                + $j <- floor(N / 2) + i$, $s_i^n <- P(i)$ e $s_j^n <- P(j)$
                                + *se* esatto *allora*
                                    + $Theta <- dstr("Bernoulli")(A(i, j)(Delta t))$
                                + *altrimenti* #comment(fill:false)[è approssimato]
                                    + $Theta <- dstr("Bernoulli")(B(i, j)(Delta t))$
                                + *se* $Theta = 1$ *allora*
                                    + $E <- E(s_i^n, i, s_j^n, j)$
                                    + $gamma <- dstr("Gamma")((1 - E)^2 / sigma^2, sigma^2 / (1 - E))$
                                    + $s_i^(n+1) <- s_i^n (1 - E + gamma)$ #comment[città interagente]
                                    + $s_j^(n+1) <- s_j^n + s_i^n E$ #comment[città ricevente]
                                + *altrimenti*
                                    + $s_i^(n+1) <- s_i^n$ e $s_j^(n+1) <- s_j^n$
                                + $cal(S)^(n+1) <- (s_1^(n+1), s_2^(n+1), ..., s_N^(n+1))$
                                + $overline(l)(s, (n+1) Delta t) <-$ istogramma di $cal(S)^(n+1)$ <algMonteCarloDistribuzioneNesima>
                        ]
                    ]
                ]
            ]
        ]
    //#endregion
//#endregion
