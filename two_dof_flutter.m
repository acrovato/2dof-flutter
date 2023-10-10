%% Aeroelasticity MATLAB session 2 corrected
% This script solves the 2D quasi-steady EoM of a flat plate (pitch and plunge)

% All units are SI unless specified
% 2D case => data in [/m]

%% Initialization
close all; clear;

% Inputs
c = 0.25; % chord
xf = 0.4*c; % flexural axis posit
t = 2e-2; % plate thickness
rhoAlu = 2700; % alu density
rhoAir = 1.225; % air density
natFreqH = 2; % natural uncoupled frequency in plunge
natFreqA = 8; % natural uncoupled frequency in pitch
u = [.1:0.1:55]; % airspeed

%% Processing
m = rhoAlu*c*t; % plate mass
S = m*(c/2 - xf);
Ia = 1/3*m*(c^2-3*c*xf+3*xf^2); % inertia around pitch
b = c/2;
Kh = (2*pi*natFreqH)^2*m; % structural stiffness in plunge
Ka = (2*pi*natFreqA)^2*Ia; % structural stiffness in pitch
e = (xf-c/4)/c; % non-dimensionnal distance between c/4 and xf

M = [m, S;S, Ia]; % mass matrix
K = [Kh,0;0,Ka]; % stiffness matrix

%% Solve
tic;
ps = zeros(length(u),2);
ks = zeros(length(u),2);
f = zeros(length(u),2);
g = zeros(length(u),2);
for i=1:length(u)
    for j = 1:2
        omega = sqrt(K(j,j)/M(j,j)); % 2*pi*natFreq
        while(1)
            % GAF matrix
            k = omega*b/u(i); % reduced frequency
            C = 1 - 0.165/(1 - 0.0455/k*1i) - 0.335/(1 - 0.3/k*1i); % Theodorsen function
            Q = [-4*pi*C*1i*k+2*pi*k^2, -2*pi*c*C-2*pi*b*1i*k-4*pi*C*(0.75*c-xf)*1i*k-2*pi*b^2*k^2;
                4*pi*e*c*C*1i*k-2*pi*(xf-c/2)*k^2, 2*pi*e*c^2*C-2*(0.75*c-xf)*pi*b*1i*k+4*pi*e*c*C*(0.75*c-xf)*1i*k+2*pi*(xf-c/2)^2*k^2+pi*b^2/4*k^2];
            % Eigenvalues solution
            p = eig((K-0.5*rhoAir*u(i)^2*Q), -M);
            p = sqrt(p);
            p(imag(p)<0) = -p(imag(p)<0);
            [~,posit] = sort(imag(p));
            p = p(posit);
            % Stopping criterion
            if (abs(imag(p(j))-omega) < 1e-4)
                ks(i,j) = k;
                ps(i,j) = p(j); % eigenvalue
                f(i,j) = imag(p(j)) / (2 * pi); % frequency
                g(i,j) = real(p(j)) / imag(p(j)); % damping
                break;
            else
                omega = imag(p(j));
            end     
        end 
    end
end
toc;

%% Plot
figure
hold on
grid on
plot(u,f(:,1), 'LineWidth', 2);
plot(u,f(:,2), 'LineWidth', 2);
legend('plunge','pitch');
xlabel('U, [m/s]', 'Fontsize', 12);
ylabel('f, [Hz]', 'Fontsize', 12);
title('Frequency evolution with airspeed', 'Fontsize', 14);

figure
hold on
grid on
plot(u,g(:,1), 'LineWidth', 2);
plot(u,g(:,2), 'LineWidth', 2);
legend('plunge','pitch');
xlabel('U, [m/s]', 'Fontsize', 12);
ylabel('g, [/]', 'Fontsize', 12);
title('Damping evolution with airspeed', 'Fontsize', 14);