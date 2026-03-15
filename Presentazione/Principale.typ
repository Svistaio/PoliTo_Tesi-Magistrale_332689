
//#region Preambolo
    #import "Preambolo.typ":* // Include i pacchetti
    // #include "Preambolo.typ" // Non include i pacchetti
    #show:slideTemplate
    #show:mathEnvTemplate
//#endregion

///

//#region Presentazione
  //#region Frontespizio
    #slide(
      margin:(
        x:4em,
        top:0em,
        bottom:0em
      ),
      extremes:true,
    )[
      #set align(center+horizon)
      
      #v(1fr)
      
      #block(
        width:120%,
        // stroke:0.5pt+CTred
      )[
        #text(
          fill:CTred,
          weight:"bold",
          size:slide-sizes.title,
          // stroke:0.5pt+CTred
        )[
          #set par(
            justify:false,
            leading:0.4em,
          )
          
          Modellizzazione della distribuzione della popolazione tra città su reti spaziali mediante la teoria cinetica dei sistemi multiagente
        ]
      ]
      
      #v(1fr)
      
      #text(
        fill:CTred,
        style:"italic",
        size:slide-sizes.sub-title,
      )[Corso di Laurea in Ingegneria Matematica]
      
      #v(1fr)
      
      #block(
        width:100%,
        height:35%,
        clip:true,
        // stroke:0.5pt+black
      )[
        #place(
          center + horizon,
          image(
            "../Figure/.fnl/Frontespizio/LogoPoliTO.jpg",
            width:7cm,
            height:auto,
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

  //#region 1° parte
    #section(title:"Introduzione")[
      #subsection(subtitle:"Legge di Zipf e Pareto")[
        #slide[
            Il primo a popolarizzare lo studio della distribuzione della popolazione è stato Zipf formulando la legge empirica
            $
              f prop 1/r quad upright("ove") cases(
                f&quad #text(font:"TeX Gyre Pagella")[ è la frequenza della parola,],
                r&quad #text(font:"TeX Gyre Pagella")[ è il suo rango nella classifica,]
              )
            $
            applicabile anche in in un contesto urbano; essa è generalizzabile mediante una distribuzione di Pareto
            $
              pzclR(s) equiv integral_s^(+oo)f_S (z)dif z
              approx c/s^beta quad
            $
            con $f_S$ è la distribuzione della popolazione, $beta in RR$ e $s gt.double 1$.
        ]
      ]

      #subsection(subtitle:"Bilognormale")[
        #slide[
          Gualandi _et al._ hanno mostrato come una bilognormale con densità
          $
            f_X (x)=xi f_X_1(x)+(1-xi)f_X_2(x),
          $
          che è una combinazione convessa di due lognormali
          $
            f_X_i (x)=1/ln(10) 1/(x sigma sqrt(2 pi)) exp(- (log x- mu)^2/(2 sigma^2)),
            quad i in {1,2},
          $
          ben adatta l'intera distribuzione della popolazione.
        ]

        #slide(
          height:36cm,
          // margin:(top:0em,bottom:0em),
          // alignment:center+horizon,
        )[
          #let img-path = "../Figure/.fnl/TikZpdf/figEsempioFunzioneLognormale.pdf"
          #includefigure((
              (path:img-path,height:14cm,dy:-0cm),
              (path:img-path,height:16.3cm,dy:-28.5cm),
            ),
            width:105%
          )
        ]
      ]

      #subsection(subtitle:"Obbiettivo")[
        #slide[
          Si vogliono simulare attraverso

            #set enum(numbering:"M1")
            + la teoria cinetica dei sistemi multiagente// (TCSMA)
            + le interazioni tra città 
            + su un grafo spaziale, 
           
          per tentare di riprodurre una distribuzione con caratteristiche analoghe a quelle realmente misurate, principalmente
          // tentando di riprodurre una distribuzione della popolazione con caratteristiche analoghe a quelle realmente misurate, principalmente
          // // Sebbene mi suoni estremamente naturale, ripassando, ho scoperto che il gerundio non può reggere una subordinata finale implicita; che strano ...
          // // https://www.treccani.it/enciclopedia/proposizioni-finali_(La-grammatica-italiana)/
          
            #set enum(numbering:"R1")
            + corpo bilognormale e
            + coda di Pareto.
           
          // Una volta raggiunto questo scopo, s'interpreta il fenomeno della migrazione tramite le leggi d'emigrazione proposte, che sono alla base delle interazioni tra città.
          // Inoltre, si sottolinea che in questo lavoro le città verranno considerate come agenti caratterizzati dalla loro popolazione, le quali interagiscono attraverso la struttura sottostante di un grafo [spaziale]; tale descrizione, salvo la rete, non è affatto dissimile a quella classica delle molecole di un gas caratterizzate dalla loro velocità e posizione. Difatti in letteratura, perlomeno quella a conoscenza dell'autore, non esistono articoli che trattano contemporaneamente la TCSMA, la distribuzione della popolazione tra città e i grafi, nonostante la rappresentazione di queste su una rete sia del tutto naturale; piú nel dettaglio
        ]
      ]
    ]

    #section(title:"Teoria dei grafi")[
      #subsection(subtitle:"Cenni")[
        #slide[
          #boxed(text:"Definizione")[
            Un grafo è un insieme $cal(G)=(cal(I),cal(E))$ formato dalla coppia dell'insieme degl'indici dei nodi $cal(I)$ e dell'insieme dei lati $cal(E)$.
          ]
          //
          #boxed(text:"Definizione")[
            Sia $N equiv |cal(I)| in NPlus$, la matrice d'adiacenza pesata $bold(M) in RR^(N times N)$ è
            $ 
              #let hsep = h(.75em)
              m_(i,j) equiv cases(
                q_(i,j)in RR&hsep" se "(i,j) in cal(E)\,,
                0&hsep" altrimenti".,
              )
            $
            // Nel caso $q_(i,j) equiv 1$ $forall i,j in cal(I)$ si ha la matrice d'adiacenza unitaria $bold(A)$.
          ]
          //
          #boxed(text:"Definizione")[
            Si definiscono la forza entrante e uscente come le quantità
            $ 
                w^-_i equiv sum_(j=1)^N m_(j,i)
                quad"e"quad
                w^+_i equiv sum_(j=1)^N m_(i,j),
                quad forall i in cal(I).
            $
            // Nel caso $q_(i,j) equiv 1$ $forall i,j in cal(I)$ si ha la matrice d'adiacenza unitaria $bold(A)$.
          ]
        ]
      ]

      #subsection(subtitle:"Tipo di grafo")[
        #slide(margin:(x:5em))[
          #boxed(text:"Ipotesi")[
            #align(center)[
              Il grafo $cal(G)$ è diretto e simmetrico ($M=M^top$).
            ]
          ]

          #boxed(text:"Ipotesi")[
            #align(center)[
              Il grafo $cal(G)$ è statico: $cal(I)$ e $cal(E)$ sono costanti nel tempo.
            ]
          ]

          #let img-path = "../Figure/.fnl/TikZpdf/figEsempioGrafoInDiretto.pdf"
          #let blk-width = 25cm
          #align(center+horizon)[
            #block(width:130%)[
              #grid(
                columns:(1fr,1fr,1fr),
                column-gutter:0em,
                stroke:draftStroke(),
                [#includefigure(
                  (path:img-path,width:7cm,dx:+8.75cm),
                  width:blk-width,draft:false,
                )],
                [#includefigure(
                  (path:img-path,width:7cm,dx:-.54cm),
                  width:blk-width,draft:false,
                )],
                [#includefigure((
                  path:img-path,width:6cm,dx:-9.4cm),
                  width:blk-width,draft:false,
                )]
              )
            ]
          ]
        ]
      ]
    ]

    #section(title:"Teorie cinetiche")[
      #subsection(subtitle:"Modello esatto")[
        #slide[
          In essenza il modello si può riassumere con tre elementi:

          #align(center)[
            #canvas({
              import draw: *

              //#region Equazioni e impostazioni
                let Ber_S = {$
                  Theta tilde DstrBernoulli(A(I,I_*) Delta t),
                  //#h(.5em) #BracketRef("Ber","S")
                $}
      
                let RI_S = {$
                  #BracketRef("RI","S")
                  cases(
                      S'_t&=S_t -S_t E(S_t,I,S_t^*,I_*)\,,
                      S_t^(*prime)&=S_t^*+S_t E(S_t,I,S_t^*,I_*)\,
                  )
                $}
      
                let AR_S-label = {$
                  #BracketRef("AR","S")
                $}
                let AR_S = {$
                  // #BracketRef("AR","S")
                  cases(
                      S_(t+Delta t)&=(1-Theta)S_t+Theta S'_t\,,
                      S^*_(t+Delta t)&=(1-Theta)S^*_t+Theta S_t^(*prime)\,,
                  )
                $}
      
                let hsep = 1em
                let vsep = 1em

                let arrow-style = (
                  stroke:(
                    thickness:2pt,
                    paint:black.lighten(70%),
                    dash:"dashed",
                  ),
                  mark:(
                    // end:">",
                    end:">>",
                    fill:black.lighten(70%),
                    scale:1.2,
                    stroke: (dash:none),
                  )
                )
              //#endregion

              //#region Scatole
                if draft {circle((0,0),radius:.25cm,fill:black,)}

                group(
                  name: "RI_S",{
                    content(
                      (0,0),
                      [#RI_S],
                      name:"text",
                      anchor:"east",
                      padding:10pt
                    )
                    rect(
                      "text.north-west",
                      "text.south-east",
                      stroke:draftStroke(),
                    )
                  }
                )
  
                group(
                  name:"Ber_S",{
                    content(
                      (rel:(hsep,0),
                      to:"RI_S.east"),
                      [#Ber_S],
                      name:"text",
                      anchor:"west",
                      padding:10pt
                    )
                    rect(
                      "text.north-west",
                      "text.south-east",
                      stroke:draftStroke(),
                    )
                  }
                )
  
                group(name:"AR_S", {
                  content(
                    (rel:(.5*hsep,-vsep),
                    to:"RI_S.south-east"),
                    [#AR_S],
                    name:"text",
                    anchor:"north",
                    padding:(left:0pt,y:10pt,right:10pt)
                  )
                  rect(
                    "text.north-west",
                    "text.south-east",
                    stroke:draftStroke(),
                  )
                })

                group(name:"AR_S-label", {
                  content(
                    (rel:(0,0),
                    to:"AR_S.west"),
                    [#AR_S-label],
                    name:"text",
                    anchor:"east",
                    padding:(right:1pt,left:10pt)
                  )
                  rect(
                    "text.north-west",
                    "text.south-east",
                    stroke:draftStroke(),
                  )
                })
              //#endregion

              //#region Frecce
                let ip = ("RI_S.south-west",20%,"RI_S.south-east") // Punto iniziale
                let fp = ("AR_S-label.north-west",50%,"AR_S-label.south-west") // Punto finale

                // let icp = (rel:(0,-3em),to:ip) // Punto di controllo iniziale
                // circle(icp,radius:.1,fill:draftFill(),stroke:draftStroke())
                // let fcp = (rel:(-2em,0),to:fp) // Punto di controllo finale
                // circle(fcp,radius:.1,fill:draftFill(),stroke:draftStroke())
                // bezier(ip,fp,icp,fcp,..arrow-style)

                let cp = (rel:(0,-3em),to:ip) // Punto di controllo
                circle(cp,radius:.1,fill:draftFill(),stroke:draftStroke())
                bezier(ip,fp,cp,..arrow-style)

                //

                let ip = ("Ber_S.south-west",75%,"Ber_S.south-east") // Punto iniziale
                let fp = ("AR_S.north-east",50%,"AR_S.south-east") // Punto finale

                // let icp = (rel:(0,-2.5em),to:ip) // Punto di controllo iniziale
                // circle(icp,radius:.1,fill:draftFill(),stroke:draftStroke())
                // let fcp = (rel:(1.5em,0),to:fp) // Punto di controllo finale
                // circle(fcp,radius:.1,fill:draftFill(),stroke:draftStroke())
                // bezier((ip,"|-","RI_S.south-east"),fp,icp,fcp,..arrow-style)

                let cp = (rel:(0,-4em),to:ip) // Punto di controllo
                circle(cp,radius:.1,fill:draftFill(),stroke:draftStroke())
                bezier(ip,fp,cp,..arrow-style)

              //#endregion
            })
          ]
           
          ove
          $
            E colon RPlus cal(I) times RPlus times cal(I) -> RPlus
          $
          è il regola d'emigrazione. // Quest'equazioni valgono anche per le singole realizzazioni.

          //#region Ipotesi TCSMA
            // Per poter applicare la TCSMA sono necessarie quattro ipotesi
  
            //   #set enum(numbering: "B1")
            //   + la distribuzione dei microstati è uniforme nello spazio;
            //   + le interazioni non binarie tra agenti sono trascurabili;
            //   + gli agenti sono indistinguibili;
            //   + due agenti che interagiscono sono indipendenti.
  
            // Tralascio queste ipotesi e non parlo della distribuzione scelta perché l'esatta forma della f non è importante per capire i risultati
          //#endregion
        ]

        #slide[
          #let LHS = $
            pderv(t)integral_cal(I) integral_(RPlus) #bis() Phi f de v de i
          $
          #let RHS = $
            integral_(cal(I)^2) integral_(RPlus^2) #bis() A(i,i_*)
            frac(chevron.l Phi' + Phi'_* - Phi - Phi_* chevron.r,2)
            f f_* de s de s_* de i de i_*
          $

          #let vphantom(body) = box(width:0pt,hide(body))
          #let vLHSpad = vphantom(RHS)
          #let vRHSpad = vphantom(LHS)

          #let circle-options = (inset:1pt,stroke:1pt)
          #let circledI = {circle(..circle-options)[#align(center)[I]]}
          #let circledII = {circle(..circle-options)[#align(center)[II]]}

          Avvalendosi dell'osservabile $Phi colon cal(I) times RPlus -> RR$ dall'#BracketRef("AR","S") si perviene a all'equazione di tipo Boltzmann
          $
            underbrace(#LHS#vLHSpad,circledI)=underbrace(#RHS#vLHSpad,circledII),
          $
          ove $f colon cal(I) times RPlus times RPlus -> RPlus$ è la distribuzione statistica del microstato $(I,S_t)$ relativo all'agente rappresentativo.

          I due membri veicolano informazioni distinte:
          
            + $circledI$ è la forma debole dell'operatore di trasporto lineare e
            + $circledII$ è la forma debole dell'operatore collisionale.
        ]
      ]

      #subsection(subtitle:"Modello approssimato")[
        #slide[
          L'approssimazione si ricava semplicemente sostituendo $bold(A)$ con
          $
            bold(B) equiv (bold(k)bold(k)^top)/D_N,
            quad text("con ") D equiv bold(1)^top bold(A)bold(1)= norm(bold(A))_1,
          $
          da cui
          $
            pderv(t)integral_(cal(I)) integral_(RPlus) #bis() Phi g de v de i =
            integral_(cal(I)^2) integral_(RPlus^2) #bis() B(i,i_*)
            frac(chevron.l Phi' + Phi'_* - Phi - Phi_* chevron.r,2)
            g g_* de s de s_* de i de i_*,
          $
          che per $N->+oo$ si può dimostrare essere equivalente a
          $
            pderv(t)integral_0^1 integral_RPlus #bis(shift:-.6em)
            phi underhat(d) de s de underhat(k)
            =integral_0^1integral_0^1integral_RPlus^2 #bis()
            (underhat(k)underhat(k,subscript:*))/(overline(D))
            (chevron.l phi'+ phi'_*- phi- phi_* chevron.r)/2
            underhat(d) underhat(d,subscript:*)
            de s de s_* de underhat(k) de underhat(k,subscript:*),
          $
          con $overline(D) equiv lim_(N->+oo)\(D_N\/N^2\)$.
        ]

        #slide(
          height:27cm,
          margin:(top:0em,bottom:0em),
          // alignment:center+horizon,
        )[
          #let img-path = "../Figure/.fnl/TikZpdf/figRappresentazioneGraficaApprossimazione.pdf"
          #includefigure((path:img-path,height:auto,dy:-0cm))
        ]
      ]
    ]
  //#endregion

  //#region 2° parte
    #section(title:"Simulazioni")[
      #subsection(subtitle:"Algoritmo")[
        // https://a5s.eu/toot-lovelace/line-numbers/line-numbers.html
        #slide(
          width:20cm,
          height:25cm,
          margin:(x:1cm,top:0cm,bottom:0cm),
          // alignment:center+horizon
        )[
            #set text(size:20pt)
            // #set align(center+horizon)

            // #let scl = 100%
            // #scale(x:scl,y:scl,origin:center+horizon)[
              #pseudocode-list(
                title:none,
                stroke:0.5pt+black,
                booktabs:true,
              )[
                - *Dati:* $N in NN^+$, $Delta t<=1$, $sigma$, $T>0$, $P$, $bold(A)$ e $bold(B)$;
                
                + #v(.5em)$pzclSsup(0)<-(s_1^0,s_2^0,...,s_N^0)equiv(P\/N)bold(1)in RPlus^N$;
                + *per* $n=0,1,2,...,floor(T\/Delta t)-1$ *fai*
                  + $p<-$ permutazione indipendente di $\{1,2,...,N\}$;
                  + *per* $c=1,2,...,floor(N\/2)$ *fai* #comment(fill:false)[numero di coppie]
                    + $i<-p(c);r<-p(floor(N\/2)+c)$;
                    + *se* esatto *allora*
                      + $Theta<-DstrBernoulli(A(i,r)Delta t)$;
                    + *altrimenti* #comment(fill:false)[è approssimato]
                      + $Theta<-DstrBernoulli(B(i,r)Delta t)$;
                    + *se* $Theta = 1$ *allora*
                      + $E<-E(s_i^n,i,s_r^n,r)$;
                      // + #comment(fill:false)[$gamma<-DstrGamma\((1-E)^2\/sigma^2,sigma^2\/(1-E)\)$;]
                      + $s_i^(n+1)<-s_i^n (1-E)$; #comment[città interagente]
                      + $s_r^(n+1)<-s_r^n+s_i^n E$; #comment[città ricevente]
                    + *altrimenti*
                      + $s_i^(n+1)<-s_i^n$; $s_r^(n+1)<-s_r^n$;
                  + $pzclSsup(n+1)<-\(s_1^(n+1),s_2^(n+1),...,s_N^(n+1)\)$;
                  + $overline(z)(s,(n+1) Delta t) <-$ istogramma di $pzclSsup(n+1)$;
              ]
            // ]
          // ]
        ]
      ]

      #subsection(subtitle:[Risultati #BracketRef("RE","S")])[
        #let img-path = "../Figure/.fnl/TikZpdf/figConfigurazioneRiferimentoInstabileRET.pdf"

        #slide[
          #grid(
            columns:(1fr,1fr),
            column-gutter:0em,
            stroke:draftStroke(),
            [
              Essa ha forma:
              
              $ E_S (s,s_*) equiv lambda frac((s_*\/s)^alpha,1 + (s_*\/s)^alpha) $
              
              con $lambda in (0,1)$ e $alpha in RR^+$. //La logica è che il tasso d'emigrazione verso la città ricevente è tanto maggiore quanto piú grande è la sua popolazione relativa, data dal rapporto $s_*\/s$, rispetto alla città interagente.
            ],
            [
              // Configurazione di riferimento:
              #set align(horizon)
              #figure(
                table(
                  columns:5,
                  stroke:none,
                  table.hline(),
                  ..padded-row(
                    ([],[$lambda$],[$alpha$],[$N_t$],[$R$],),
                    pos:bottom,
                    pad:.3em
                  ),
                  table.hline(),
                  ..padded-row(([Stabile],[$0.1$],[$0.5$],[$10^7$],[$100$])),
                  [Instabile],[$0.1$],[$1.5$],[$10^6$],[$10$],
                  table.hline(),
                ),// caption:[],
                kind:table,
              ) //<tabParametriRegolaEmigrazioneTaglia>
            ],
          )

          #v(1fr)
          #includefigure(
            width:120%,
            (path:img-path,height:rowHeight*factor(1.8,-1),dy:-30cm)
          )
          #v(1fr)
        ]

        #slide(
          margin:(top:0em,bottom:0em),
          // alignment:center+horizon
        )[
          #includefigure((
            (path:img-path,height:rowHeight*factor(5.5,-2),dy:-0cm),
            (path:img-path,height:img-height-14,dy:img-dy(shift:-.65cm)),
          ))
        ]
      ]

      #subsection(subtitle:[Risultati #BracketRef("RE","SK")])[
        #let img-path = "../Figure/.fnl/TikZpdf/figConfigurazioneRiferimentoRETG.pdf"

        #slide[
          #grid(
            columns:(2fr,1fr),
            column-gutter:0em,
            stroke:draftStroke(),
            [
              Essa ha forma:
              
              $ E_(S K)(s,i,s_*,i_*) equiv lambda frac([(s_*\/s)(k_(i_*)\/k_i)]^alpha,1 + [(s_*\/s)(k_(i_*)\/k_i)]^alpha) $
              
              con $lambda in (0,1)$ e $alpha in RR^+$. //La logica è che il tasso d'emigrazione verso la città ricevente è tanto maggiore quanto piú grande è la sua popolazione relativa, data dal rapporto $s_*\/s$, rispetto alla città interagente.
            ],
            [
              // Configurazione di riferimento:
              #set align(horizon)
              #figure(
                table(
                  columns:4,
                  stroke:none,
                  table.hline(),
                  ..padded-row(
                    ([$lambda$],[$alpha$],[$N_t$],[$R$],),
                    pos:bottom,
                    pad:.3em
                  ),
                  table.hline(),
                  ..padded-row(([$0.15$],[$0.63$],[#num(e:6)[3]],[$100$])),
                  table.hline(),
                ),// caption:[],
                kind:table,
              ) //<tabParametriRegolaEmigrazioneTaglia>
            ],
          )

          #v(1fr)
          #includefigure(
            width:120%,
            (path:img-path,height:rowHeight*factor(1.7,-1),dy:-29.4cm)
          )
          #v(1fr)
        ]

        #slide(
          // alignment:center+horizon
        )[
          #includefigure((
            (path:img-path,height:rowHeight,dy:-0cm),
            (path:img-path,height:img-height-14,dy:img-dy()),
          ))
        ]

        #slide(
          // alignment:center+horizon
        )[
          #includefigure((
            path:img-path,height:rowHeight*factor(2,-2,base:2),dy:-6.45cm
          ))
        ]
      ]

      #subsection(subtitle:"Sulla convergenza")[
        #slide(
          // height:27cm,
          // margin:(top:0em,bottom:0em),
        )[
          La convergenza della #BracketRef("RE","SK") non è una proprietà scontata:

          #block(
            width:100%,
            below:1.75em,
            above:1.75em
          )[
            #set align(center)
            #canvas({
              import draw: *

              //#region Impostazioni
                let vsep = 1em
                let hsep = 2.5em

                let dsep = 20deg
                let radius = .6em

                let arrow-style = (
                  stroke:(
                    thickness:1pt,
                    paint:black,
                    // dash:"dashed",
                  ),
                  mark:(
                    end:">",
                    // end:">>",
                    fill:black,
                    scale:1,
                    // stroke: (dash:none),
                  )
                )

                let lsep = 1.5em
                let letter-size = 1.2*radius

                let tsep = 1.75em
                let fsep = 3em
                let text-size = .8*slide-sizes.text
              //#endregion

              //#region Nodi
                if draft {circle((0,0),radius:.1cm,fill:black)}

                circle((-hsep,0),radius:radius,name:"vi")
                circle((hsep,0),radius:radius,name:"vr")
              //#endregion

              //#region Frecce
                let mp = ("vi",50%,"vr") // Punto medio

                // Freccia sinistra
                let cpl = (rel:(0,-vsep),to:mp) // Punto di controllo
                bezier(
                  (name:"vr",anchor:180deg+dsep),
                  (name:"vi",anchor:-dsep),
                  cpl,..arrow-style
                )
                circle(
                  cpl,radius:.1,
                  fill:draftFill(colour:red),
                  stroke:draftStroke(colour:red)
                )

                // Freccia destra
                let cpr = (rel:(0,vsep),to:mp) // Punto di controllo
                bezier(
                  (name:"vi",anchor:+dsep),
                  (name:"vr",anchor:180deg-dsep),
                  cpr,..arrow-style
                )
                circle(
                  cpr,radius:.1,
                  fill:draftFill(colour:red),
                  stroke:draftStroke(colour:red)
                )
              //#endregion

              //#region Testo
                //#region Nodi relativi
                  content(
                    (rel:(-tsep,0),to:"vi"),
                    text(size:text-size)[
                      #set align(center)
                      #set par(leading:0.2em)
                      Nodo int.\ rel. maggiore
                    ],
                    anchor:"east"
                  )
                  content(
                    (rel:(+tsep,0),to:"vr"),
                    text(size:text-size)[
                      #set align(center)
                      #set par(leading:0.2em,)
                      Nodo ric.\ rel. minore
                    ],
                    anchor:"west"
                  )
                //#endregion

                //#region Simboli interagente
                  content(
                    "vi",
                    text(size:letter-size)[$i$],
                    // anchor:"west"
                  )
                  content(
                    (rel:(0,lsep),to:"vi"),
                    text(size:letter-size)[$s$],
                    // anchor:"west"
                  )
                  content(
                    (rel:(0,-lsep),to:"vi"),
                    text(size:letter-size)[$k$],
                    // anchor:"west"
                  )
                //#endregion

                //#region Simboli ricevente
                  content(
                    "vr",
                    text(size:letter-size)[$i_*$],
                    // anchor:"west"
                  )
                  content(
                    (rel:(0,lsep),to:"vr"),
                    text(size:letter-size)[$s_*$],
                    // anchor:"west"
                  )
                  content(
                    (rel:(0,-lsep),to:"vr"),
                    text(size:letter-size)[$k_*$],
                    // anchor:"west"
                  )
                //#endregion

                //#region Maggiore
                  // https://forum.typst.app/t/how-do-i-use-the-rotate-function-in-the-cetz-content-environment/3055
                  content(
                    (0,-lsep),//angle:+90deg,
                    text(size:letter-size)[>]
                  )
                  content(
                    (0,+lsep),//angle:+90deg,
                    text(size:letter-size)[>]
                  )
                //#endregion

                //#region Flussi
                  content(
                    (0,-fsep),//angle:+90deg,
                    text(size:text-size)[
                      Flusso accentrante
                    ],anchor:"south"
                  )
                  content(
                    (0,+fsep),//angle:+90deg,
                    text(size:text-size)[
                      Retroflusso stabilizzante
                    ],anchor:"north"
                  )
                //#endregion
              //#endregion
            })
          ]

          Infatti, dalla funzione di Hill $H^alpha (r)equiv lambda (r^alpha)/(1+r^alpha)$, vale
          $
            r_(S K)=(s_*)/s (k_*)/k<(s_*)/s=r_S
            quad ==> quad
            H^alpha (r_(S K))< H^alpha (r_S),
          $
          poiché $H^alpha (r)$ è una funzione crescente.
        ]
      ]

      // #subsection(subtitle:"È ragionevole?")[
      //   #slide(
      //     // height:27cm,
      //     // margin:(top:0em,bottom:0em),
      //   )[
      //     Sí, per due vincoli: uno di natura fisica e l'altro sociale.
      //     #let img-path = "../Figure/.fnl/TikZpdf/figForzaVsGrado20.pdf"
      //     #includefigure((path:img-path),width:100%)
      //   ]
      // ]

      #subsection(subtitle:[Risultati #BracketRef("RE","SW")])[
        #let img-path = "../Figure/.fnl/TikZpdf/figConfigurazioneRiferimentoRETF.pdf"

        #slide[
          #grid(
            columns:(2fr,1fr),
            column-gutter:0em,
            stroke:draftStroke(),
            [
              Essa ha forma:
              
              $ E_(S K)(s,i,s_*,i_*) equiv lambda frac([(s_*\/s)(w_(i_*)\/w_i)]^alpha,1 + [(s_*\/s)(k_(i_*)\/k_i)]^alpha) $
              
              con $lambda in (0,1)$ e $alpha in RR^+$. //La logica è che il tasso d'emigrazione verso la città ricevente è tanto maggiore quanto piú grande è la sua popolazione relativa, data dal rapporto $s_*\/s$, rispetto alla città interagente.
            ],
            [
              // Configurazione di riferimento:
              #set align(horizon)
              #figure(
                table(
                  columns:4,
                  stroke:none,
                  table.hline(),
                  ..padded-row(
                    ([$lambda$],[$alpha$],[$N_t$],[$R$],),
                    pos:bottom,
                    pad:.3em
                  ),
                  table.hline(),
                  ..padded-row(([$0.18$],[$0.44$],[#num(e:6)[3]],[$100$])),
                  table.hline(),
                ),// caption:[],
                kind:table,
              ) //<tabParametriRegolaEmigrazioneTaglia>
            ],
          )

          #v(1fr)
          #includefigure(
            width:120%,
            (path:img-path,height:rowHeight*factor(1.7,-1),dy:-29.4cm)
          )
          #v(1fr)
        ]

        #slide(
          // alignment:center+horizon
        )[
          #includefigure((
            (path:img-path,height:rowHeight,dy:-0cm),
            (path:img-path,height:img-height-14,dy:img-dy()),
          ))
        ]

        #slide(
          // alignment:center+horizon
        )[
          #includefigure((
            path:img-path,height:rowHeight*factor(2,-2,base:2),dy:-6.45cm
          ))
        ]
      ]

    ]

    #section(title:"Interpretazione")[
      #subsection(subtitle:"Paremetri")[
        #slide(
          // height:27cm,
          // margin:(top:0em,bottom:0em),
          // alignment:center+horizon,
        )[
          Da alcuni studi parametrici, $lambda$ e $alpha$ sono cosí interpretabili:

            - $lambda$ è l'attrazione esercitata dai centri relativamente maggiori e
            - $alpha$ è la repulsione sprigionata dai centri relativamente minori.
            // essa è chiaramente inversamente legata all'entità del retroflusso stabilizzante.

          #v(1fr)

          #let img-path = "../Figure/.fnl/TikZpdf/figFunzioneHillGradoa.pdf"
          #includefigure((path:img-path,height:auto,dy:-0cm),width:80%)

          #v(1fr)
        ]
      ]
    ]

    #section(title:"Conclusioni")[
      #subsection(subtitle:"Sviluppi futuri")[
        #slide(
          alignment:top,
        )[
          Il lavoro non è esaustivo e conduce a molteplici sviluppi futuri:

            1. definire regole d'interazione piú complesse,
            2. ricavare l'equazione di Fokker-Planck,
            3. considerare una rete dinamica coevolvente e
            4. applicare la teoria a reti europee o internazionali.
        ]
      ]
    ]
  //#endregion

  //#region Ringraziamenti
    #slide(
      margin:(
        x:0em,
        top:0em,
        bottom:0em
      ),
      extremes:true,
    )[
      #set align(center+horizon)
      
      #block(
        width:120%,
        // stroke:0.5pt+CTred
      )[
        #text(
          fill:CTred,
          // weight:"bold",
          size:slide-sizes.title+5pt,
          // stroke:0.5pt+CTred
        )[Grazie dell'attenzione]
      ]
    ]
  //#endregion
//#endregion
