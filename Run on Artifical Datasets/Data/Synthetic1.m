function data = outlier(N, r, dist, outliers)

    if nargin < 1
        N = 600;
    end
    if nargin < 2
        r = 20;
    end
    if nargin < 3
        dist = 30;
    end
    if nargin < 4
        outliers = 0.04;
    end

    N1 = round(N * (.5-outliers));
    N2 = N1;

    phi1 = rand(N1,1) * pi;
    r1 = sqrt(rand(N1,1))*r;
    P1 = [-dist + r1.*sin(phi1) r1.*cos(phi1) ones(N1,1)];

    phi2 = rand(N2,1) * pi;
    r2 = sqrt(rand(N2,1))*r;
    P2 = [dist - r2.*sin(phi2) r2.*cos(phi2) 2*ones(N2,1)];    
    
    data = [P1; P2];

end