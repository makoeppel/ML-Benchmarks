subroutine random_matrix(n, m, a)
    implicit none

    ! read in arguments
    integer, intent(in) :: n
    integer, intent(in) :: m
    double precision, intent(out) :: a(n, m)
    double precision :: r
    integer :: i, j

    do i = 1, n
        do j = 1, m
            call random_number(r)
            a(i, j) = r
        end do
    end do
end subroutine random_matrix


subroutine squard_error(n, a, y, error)
    implicit none

    ! read in arguments
    integer, intent(in) :: n
    double precision, intent(in) :: a(n, 1)
    double precision, intent(in) :: y(n, 1)
    double precision, intent(out) :: error

    integer :: i
    double precision :: errors = 0

    do i = 1, n
        ! print *, y(i, 1), a(i, 1)
        errors = errors + (a(i, 1) - y(i, 1))**2
    end do
    error = errors / n
    errors = 0
end subroutine squard_error


subroutine add(n, m, a, v)
    implicit none

    ! read in arguments
    integer, intent(in) :: n
    integer, intent(in) :: m
    double precision, intent(in) :: v(1, m)
    double precision, intent(out) :: a(n, m)

    integer :: i, j

    do i = 1, n
        do j = 1, m
            ! print *, v(j, 1), i, j
            a(i, j) = a(i, j) + v(j, 1)
        end do
    end do

end subroutine add


subroutine sub(n, m, a, v)
    implicit none

    ! read in arguments
    integer, intent(in) :: n
    integer, intent(in) :: m
    double precision, intent(in) :: v(1, m)
    double precision, intent(out) :: a(n, m)

    integer :: i, j

    do i = 1, n
        do j = 1, m
            ! print *, v(i, 1), i, j
            a(i, j) = a(i, j) - v(i, 1)
        end do
    end do

end subroutine sub


subroutine build_mlp(w1n, w1m, w1, w2n, w2m, w2, b1n, b1m, b1, b2n, b2m, b2)
    implicit none
    
    ! read in arguments
    integer, intent(in) :: w1n
    integer, intent(in) :: w1m
    integer, intent(in) :: w2n
    integer, intent(in) :: w2m
    integer, intent(in) :: b1n
    integer, intent(in) :: b1m
    integer, intent(in) :: b2n
    integer, intent(in) :: b2m
    double precision, intent(out) :: w1(w1n, w1m)
    double precision, intent(out) :: w2(w2n, w2m)
    double precision, intent(out) :: b1(b1n, b1m)
    double precision, intent(out) :: b2(b2n, b2m)

    call random_matrix(w1n, w1m, w1)
    call random_matrix(w2n, w2m, w2)
    call random_matrix(b1n, b1m, b1)
    call random_matrix(b2n, b2m, b2)
end subroutine build_mlp


subroutine get_xor_data(xn, xm, x, yn, ym, y, n, m)
    implicit none
    
    ! read in arguments
    integer, intent(in) :: xn      ! in means read value
    integer, intent(in) :: xm      ! inout means read-write 
    double precision, intent(out) :: x(xn, xm) ! out means write value

    integer, intent(in) :: yn
    integer, intent(in) :: ym
    double precision, intent(out) :: y(yn, ym)

    integer, intent(out) :: n
    integer, intent(out) :: m

    x = reshape((/ 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0 /), shape(x))
    y = reshape((/ 0.0, 1.0, 1.0, 0.0 /), shape(y))
    n = 2
    m = 4
end subroutine get_xor_data

subroutine print_matrix(n, m, a)
    implicit none

    integer, intent(in) :: n
    integer, intent(in) :: m
    double precision, intent(in) :: a(n, m)
  
    integer :: i
  
    do i = 1, n
      print *, a(i, 1:m)
    end do
  
end subroutine print_matrix

program mlp
    use, intrinsic :: iso_c_binding, only: sp=>c_float, dp=>c_double
    implicit none

    ! define values and variables
    character(100) :: data_path
    character(100) :: num2char
    character(100) :: num3char
    character(100) :: num4char
    character(100) :: num5char
    character(100) :: num6char
    character(100) :: num7char
    character(100) :: num8char

    integer :: layer0
    integer :: layer1
    integer :: layer2
    real(dp) :: learning_rate
    integer :: epoch
    integer :: seed
    logical :: test

    double precision, dimension (:,:), allocatable :: x!(2, 4)
    double precision, dimension (:,:), allocatable :: y!(1, 4)
    integer :: n, i, m
    double precision :: error
    
    double precision, dimension (:,:), allocatable :: w1!(2, 3)
    double precision, dimension (:,:), allocatable :: w2!(3, 1)
    double precision, dimension (:,:), allocatable :: b1!(1, 3)
    double precision, dimension (:,:), allocatable :: b2!(1, 1)
    double precision, dimension (:,:), allocatable :: z1!(4, 3)
    double precision, dimension (:,:), allocatable :: z2!(4, 1)
    double precision, dimension (:,:), allocatable :: s1!(4, 3)
    double precision, dimension (:,:), allocatable :: s2!(4, 1)
    double precision, dimension (:,:), allocatable :: delta1!(4, 3)
    double precision, dimension (:,:), allocatable :: delta2!(4, 1)
    double precision, dimension (:,:), allocatable :: gradients1!(2, 3)
    double precision, dimension (:,:), allocatable :: gradients2!(3, 1)

    ! check command line arguments
    if ( command_argument_count().NE.8 ) then
        print *, "Usage: "
        print *, "mlp <data-path> <layer0> <layer1> <layer2> <learning-rate> <epoch> <seed> <test>"
        stop 0
    endif

    ! read in command line arguments
    call get_command_argument(1, data_path)
    call get_command_argument(2, num2char)
    call get_command_argument(3, num3char)
    call get_command_argument(4, num4char)
    call get_command_argument(5, num5char)
    call get_command_argument(6, num6char)
    call get_command_argument(7, num7char)
    call get_command_argument(8, num8char)

    ! convert to correct type
    read(num2char,*)layer0
    read(num3char,*)layer1
    read(num4char,*)layer2
    read(num5char,*)learning_rate
    read(num6char,*)epoch
    read(num7char,*)seed
    if ( num8char == "true" ) then
        test = .true.
        print *, '##### Create Dataset #####'
        allocate ( x(4, 2) )
        allocate ( y(4, 1) )
        call get_xor_data(4, 2, x, 4, 1, y, n, m)
    else
        test = .false.
    endif

    ! print values
    print *, 'x:'
    call print_matrix(4, 2, x)
    print *, 'y:'
    call print_matrix(4, 1, y)

    print *, '##### Build Network #####'
    call random_seed(seed)
    allocate ( w1(2, layer2) )
    allocate ( w2(layer2, 1) )
    allocate ( b1(1, layer2) )
    allocate ( b2(1, 1) )
    allocate ( z1(m, layer2) )
    allocate ( z2(m, 1) )
    allocate ( s1(m, layer2) )
    allocate ( s2(m, 1) )
    call build_mlp(2, layer2, w1, layer2, 1, w2, 1, layer2, b1, 1, 1, b2)
    print *, 'w1:'
    call print_matrix(2, layer2, w1)
    print *, 'w2:'
    call print_matrix(layer2, 1, w2)
    print *, 'b1:'
    call print_matrix(1, layer2, b1)
    print *, 'b2:'
    call print_matrix(1, 1, b2)

    print *, '##### Train Network #####'
    do i = 1, epoch        
        ! forward
        ! (x.T x w) + b
        z1 = matmul(x, w1)
        call add(4, layer2, z1, b1)
        s1 = 1. / (1. + exp(-z1))

        z2 = matmul(s1, w2)
        call add(4, 1, z2, b2)
        s2 = 1. / (1. + exp(-z2))

        ! error
        call squard_error(4, s2, y, error)
        print *, "Epoch: ", i, " Error: ", error
        
        ! backwards
        delta2 = s2
        call sub(4, 1, delta2, y)
        delta2 = delta2 * s2 * (1 - s2)
        gradients2 = matmul(transpose(s1), delta2)

        delta1 = matmul(delta2, transpose(w2)) * s1 * (1 - s1)
        gradients1 = matmul(transpose(x), delta1)

        w2 = w2 - learning_rate * gradients2
        w1 = w1 - learning_rate * gradients1
        b2 = b2 - learning_rate * sum(delta2)
        b1 = b1 - learning_rate * sum(delta1)

    end do

    call print_matrix(4, 1, s2)
    call print_matrix(4, 1, y)

end program mlp