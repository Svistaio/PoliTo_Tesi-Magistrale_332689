function funmat2tex(matpath,texpath)
    s = load(matpath);
    f = string(fieldnames(s.plots));

    figure("Visible","off");
    hold on

    for i = 1:numel(f)
        field = f(i);

        switch s.plots.(field).t
            case 'scatter'
                scatter( ...
                    s.plots.(field).x, ...
                    s.plots.(field).y, ...
                    'DisplayName', s.plots.(field).l ...
                );

            case 'function'
                plot( ...
                    s.plots.(field).x,s.plots.(field).y, ...
                    'DisplayName',s.plots.(field).l ...
                );

            case 'histogram'
        end
    end

    grid on
    ax = gca;
    ax.GridLineStyle = '--';

    xlabel( ...
        s.style.labels.x,...
        'interpreter','latex' ...
    );
    ylabel( ...
        s.style.labels.y,...
        'interpreter','latex' ...
    );

    set(ax,'XScale',s.style.scale.x)
    set(ax,'YScale',s.style.scale.x)

    % legend( ...
    %     'Location','best',...
    %     'interpreter','latex' ...
    % );

    matlab2tikz(texpath,'standalone',true);

end

matpath = "..\Figure\.mat\20\NA\AAssortativity.mat";
texpath = "..\Figure\.tex\Prova.tex";
funmat2tex(matpath,texpath)
